import gym
import time
import numpy as np
import os.path as osp
import tensorflow as tf
import baselines.common.tf_util as U

from tqdm import tqdm
from mpi4py import MPI
from baselines.common import fmt_row
from baselines.common.td3_util import * 
from baselines import logger, bench
from baselines.common.misc_util import boolean_flag

from baselines.common.mpi_adam import MpiAdam
from baselines.imitation.d2_discriminator import D2Discriminator
from baselines.imitation.expert_object import ExpertObject

def _update_pbar_msg(pbar, total_timesteps):
  """Update the progress bar with the current training phase."""
  if total_timesteps < int(1e3):
    msg = 'not training'
  else:
    msg = 'training'
  if total_timesteps < int(2e3):
    msg += ' rand acts'
  else:
    msg += ' policy acts'
  if pbar.desc != msg:
    pbar.set_description(msg)

def D2Imitation(env_id, expert_path, use_discriminator, min_target, repeat_traj, 
        hidden_size, prob_threshold, expert_sampling, 
        actor_critic=mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        total_timesteps=int(1e5), replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-4, q_lr=1e-3, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, eval_every=1000, act_noise=0.1, 
        target_noise=0.2, noise_clip=0.5, policy_delay=2, max_ep_len=1000):

    env_fn = lambda : gym.make(args.env_id)

    expert_obj = ExpertObject(env_id)
    expert_obj.load_traj_from_files(expert_path, repeat_trajs=repeat_traj)

    sa_discriminator = D2Discriminator(env_fn(), hidden_size)
    d_adam = MpiAdam(sa_discriminator.get_trainable_variables())
    d_step = 2000
    d_stepsize=3e-4

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, _ = env_fn(), env_fn()
    max_ep_len = env._max_episode_steps

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), ""))

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_limit = env.action_space.high[0]
    ac_kwargs['action_space'] = env.action_space

    x_ph, a_ph, x2_ph, r_ph, d_ph = placeholders(obs_dim, act_dim, obs_dim, None, None)
    pi_lr_ph, q_lr_ph = tf.placeholder(tf.float32, []), tf.placeholder(tf.float32, [])

    with tf.variable_scope('main'):
        pi, q1, q2, q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    with tf.variable_scope('target'):
        pi_targ, _, _, _  = actor_critic(x2_ph, a_ph, **ac_kwargs)
    
    with tf.variable_scope('target', reuse=True):
        epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)
        _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    if min_target:
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*min_q_targ)

        q1_loss = tf.reduce_mean((q1-backup)**2)
        q2_loss = tf.reduce_mean((q2-backup)**2)
        q_loss = q1_loss + q2_loss
    else:
        min_q_targ = q1_targ
        backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*min_q_targ)

        q1_loss = tf.reduce_mean((q1-backup)**2)
        q_loss = q1_loss

    pi_loss = -tf.reduce_mean(q1_pi)

    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr_ph)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr_ph)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    nworkers = MPI.COMM_WORLD.Get_size()

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    d_adam.sync()

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    logger.log(fmt_row(16, sa_discriminator.loss_name))
    for _ in range(d_step):
        ob_expert, ac_expert = expert_obj.sample_sa(batch_size)
        ob_expert, ac_expert = np.array(ob_expert), np.array(ac_expert)
        ac_random = np.array([env.action_space.sample() for _ in range(batch_size)])
        *newlosses, g = sa_discriminator.lossandgrad(ob_expert, ac_random, ac_expert)
        d_adam.update(allmean(g), d_stepsize)
    logger.log(fmt_row(16, newlosses))

    # Main loop: collect experience in env and update/log each epoch
    with tqdm(total=total_timesteps, desc='') as pbar:
        for t in range(total_timesteps):
            _update_pbar_msg(pbar, t)

            if t > start_steps:
                a = get_action(o, act_noise)
            else:
                a = env.action_space.sample()

            # Step the env
            o2, r, d, info = env.step(a)
            ep_ret += r
            ep_len += 1

            d = False if ep_len==max_ep_len else d

            # Store experience to replay buffer
            if t > start_steps and use_discriminator:
                prob = sa_discriminator.get_prob(o, a)[0]
                if prob > prob_threshold:
                    expert_obj.feed_one_transition(o, a, o2, r, d, info)
                else:
                    replay_buffer.store(o, a, r, o2, d)
            else:
                replay_buffer.store(o, a, r, o2, d)

            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                o, ep_ret, ep_len = env.reset(), 0, 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                # update critics

                for j in range(update_every):
                    if expert_sampling == 'expert':
                        ex_obs1, ex_acts, ex_obs2, _, ex_done, _ = expert_obj.sample(batch_size)
                    elif expert_sampling == 'mixed':
                        ex_obs1, ex_acts, ex_obs2, _, ex_done, _ = expert_obj.sample_mixed(batch_size)
                    else:
                        raise NotImplementedError

                    ex_obs1, ex_acts, ex_obs2 = np.array(ex_obs1), np.array(ex_acts), np.array(ex_obs2)

                    ex_rews = np.ones_like(ex_done)

                    batch = replay_buffer.sample_batch(batch_size)
                    obs1, acts, obs2, done = batch['obs1'], batch['acts'], batch['obs2'], batch['done']
                    rews = np.zeros_like(done)

                    obs1 = np.concatenate((obs1, ex_obs1), 0)
                    obs2 = np.concatenate((obs2, ex_obs2), 0)
                    acts = np.concatenate((acts, ex_acts), 0)
                    done = np.concatenate((done, ex_done), 0)
                    rews = np.concatenate((rews, ex_rews), 0).flatten()

                    feed_dict = {x_ph: obs1,
                                x2_ph: obs2,
                                a_ph: acts,
                                r_ph: rews,
                                d_ph: done,
                                pi_lr_ph: pi_lr,
                                q_lr_ph: q_lr }
                    q_step_ops = [q_loss, q1, q2, train_q_op]
                    sess.run(q_step_ops, feed_dict)

                    if j % policy_delay == 0:
                        # Delayed policy update
                        sess.run([pi_loss, train_pi_op, target_update], feed_dict)

            if t % eval_every == 0:
                policy_fn = lambda ob: get_action(ob, 0.0)
                eval_len, eval_ret = [], []
                for _ in range(10):
                    eval_rews = expert_obj.eval_expert(policy_fn, render=False)
                    eval_len.append(len(eval_rews))
                    eval_ret.append(sum(eval_rews))
                logger.record_tabular('EvalEpRetMin', np.min(eval_ret))
                logger.record_tabular('EvalEpRetMax', np.max(eval_ret))
                logger.record_tabular('EvalEpRetMean', np.mean(eval_ret))
                logger.record_tabular('EvalEpLenMin', np.min(eval_len))
                logger.record_tabular('EvalEpLenMax', np.max(eval_len))
                logger.record_tabular('EvalEpLenMean', np.mean(eval_len))
                logger.record_tabular('TotalEnvInteracts', t)
                logger.record_tabular('Time', time.time()-start_time)
                logger.dump_tabular()
            pbar.update(1)

def get_task_name(args):
    task_name = 'D2'
    task_name += '.{}.'.format(args.env_id)
    task_name += "%s"%args.expert_path.split('/')[-1]
    task_name += ".prob_%s"%args.prob_threshold
    task_name += ".seed_{}".format(args.seed)
    return task_name

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v2')
    parser.add_argument('--expert_path', type=str, default=None)
    parser.add_argument('--prob_threshold', type=float, default=0.95)
    parser.add_argument('--expert_sampling', type=str, default='mixed')
    boolean_flag(parser, 'use_discriminator', default=True)
    boolean_flag(parser, 'min_target', default=True)
    boolean_flag(parser, 'repeat_traj', default=True)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--total_timesteps', type=int, default=int(5e5))
    args = parser.parse_args()
    assert args.expert_sampling in ['expert', 'mixed']

    task_name = get_task_name(args)
    logger.configure(dir='log/%s'%task_name)

    D2Imitation(args.env_id, expert_path=args.expert_path, 
        use_discriminator=args.use_discriminator, min_target=args.min_target, 
        repeat_traj=args.repeat_traj, hidden_size=args.hid, 
        prob_threshold=args.prob_threshold, 
        expert_sampling=args.expert_sampling, 
        actor_critic=mlp_actor_critic, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, total_timesteps=args.total_timesteps)
