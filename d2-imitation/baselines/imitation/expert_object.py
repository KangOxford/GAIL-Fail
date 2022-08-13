import copy
import random
import numpy as np
from gym import make

from pathlib import Path

class ExpertObject:
    def __init__(self, env_id, max_num_trajs=10000, dir_samples=None):
        self.idx = 0
        self.max_num_trajs = max_num_trajs
        self.env_id = env_id
        self.env = make(self.env_id)
        self.init_state = self.env.reset()
        # self.act_limit = self.env.action_space.high[0]
        self.picked_trajs = None
        self.trajs = []
        self.traj_rets = []
        self.traj_tags = []
        self.meta_info = []
        self.transitions = []
        self.transitions_sampled = []
        self.transition_tags = []
        self.oracle_fn = None
        self.dir_samples = None
        self.traj_filename = None
        if dir_samples != None:
            self.dir_samples = dir_samples
            Path(self.dir_samples).mkdir(parents=True, exist_ok=True)

    @property
    def num_trajs(self):
        return len(self.trajs)

    @property
    def num_transitions(self):
        return len(self.transitions)

    def get_trajs(self):
        return copy.deepcopy(self.trajs)

    def set_oracle(self, oracle_fn):
        self.oracle_fn = oracle_fn

    def set_trajs(self, demo_trajs, demo_traj_rets, traj_tags):
        if len(demo_trajs) > self.max_num_trajs:
            self.trajs = copy.deepcopy(demo_trajs[:self.max_num_trajs])
            self.traj_rets = copy.deepcopy(demo_traj_rets[:self.max_num_trajs])
            self.traj_tags = copy.deepcopy(traj_tags[:self.max_num_trajs])
        else:
            self.trajs = copy.deepcopy(demo_trajs)
            self.traj_rets = copy.deepcopy(demo_traj_rets)
            self.traj_tags = copy.deepcopy(traj_tags)
        self.make_transitions()

    def get_transitions(self):
        return copy.deepcopy(self.transitions)

    def get_sasa_transitions(self):
        return copy.deepcopy(self.transitions)

    def clear(self):
        self.trajs = []
        self.traj_rets = []
        self.traj_tags = []
        self.transitions = []

    def feed(self, states, actions, next_states, rewards, dones, infos):
        traj = copy.deepcopy(list(zip(states, actions, next_states, rewards, dones, infos)))
        traj_ret = sum(rewards)
        self.trajs.append(traj)
        self.traj_rets.append(traj_ret)
        self.transitions += traj

    def feed_one_transition(self, s, a, n_s, r, d, info):
        tran = copy.deepcopy((s, a, n_s, r, d, info))
        self.transitions_sampled.append(tran)
        if len(self.transitions_sampled) > len(self.transitions)//4:
            self.transitions_sampled.pop(0)
    
    def load(self, filename):
        if self.dir_samples:
            filename = "%s/%s"%(self.dir_samples, filename)
        data = np.load(filename)
        self.trajs = data['trajs']
        self.traj_rets = data['traj_rets']
        self.traj_tags = data['traj_tags']
        self.make_transitions()

    def load_gail_dataset(self, filename):
        traj_data = np.load(filename, allow_pickle=True)
        obs = traj_data['obs'][:]
        acs = traj_data['acs'][:]
        rets = traj_data['ep_rets']

        self.trajs = [list(zip(obs[i], acs[i])) for i in range(len(obs))]
        self.traj_rets = rets
        self.make_transitions()

        from baselines import logger
        self.num_traj = len(self.trajs)
        self.num_transition = len(self.transitions)
        self.avg_ret = sum(self.traj_rets)/len(self.traj_rets)
        self.std_ret = np.std(np.array(self.traj_rets))

        logger.log('Load GAIL mujoco dataset')
        logger.log("Total trajectories: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def load_traj_from_files(self, filename, expert_data_folder='expert_data', repeat_trajs=False):
        self.traj_filename = filename.split("/")[-1]
        if 'pure' in filename:
            expert_data_folder += '-pure'
        elif '-v2' in self.env_id:
            expert_data_folder += '-v2'
        elif '-v3' in self.env_id:
            expert_data_folder += '-v3'
        else:
            expert_data_folder += '-classic'

        print('Loading expert data...')
        with open(filename) as fin:
            line = fin.readline().strip() # remove \n
            while line:
                if self.env_id not in line:
                    line = fin.readline().strip() # remove \n
                    continue
                traj_file = "%s/%s"%(expert_data_folder, line)
                data = np.load(traj_file, allow_pickle=True)
                trajs, traj_rets, meta = data['trajs'], data['traj_rets'], data['meta']
                meta = meta.flatten()[0]
                meta_env_id, meta_info = meta['env_id'], meta['info']

                self.trajs.append(trajs.tolist())
                self.traj_tags.append(meta_info)
                traj_rets = traj_rets.flatten()[0]
                self.traj_rets.append(traj_rets)
                self.meta_info.append(meta_info)

                line = fin.readline().strip() # remove \n

        # self.plot_return_hist()
        if repeat_trajs:
            self.make_transitions_with_repeated_trajs()
        else:
            self.make_transitions()
        print('Expert data loaded.')

    def load_traj_from_files_valuedice(self, filename, num_trajectories):
        with open(filename, 'rb') as fin:
            expert_data = np.load(fin)
            expert_data = {key: expert_data[key] for key in expert_data.files}

            expert_states = expert_data['states']
            expert_actions = expert_data['actions']
            expert_next_states = expert_data['next_states']
            expert_dones = expert_data['dones']

        expert_states_traj = [[]]
        expert_actions_traj = [[]]
        expert_next_states_traj = [[]]
        expert_dones_traj = [[]]

        # states, actions, next_states, rewards, dones, infos = zip(*traj)

        for i in range(expert_states.shape[0]):
            expert_states_traj[-1].append(expert_states[i])
            expert_actions_traj[-1].append(expert_actions[i])
            expert_next_states_traj[-1].append(expert_next_states[i])
            expert_dones_traj[-1].append(expert_dones[i])

            if expert_dones[i] and i < expert_states.shape[0] - 1:
                expert_states_traj.append([])
                expert_actions_traj.append([])
                expert_next_states_traj.append([])
                expert_dones_traj.append([])

        shuffle_inds = list(range(len(expert_states_traj)))
        random.shuffle(shuffle_inds)
        shuffle_inds = shuffle_inds[:num_trajectories]
        expert_states_traj = [expert_states_traj[i] for i in shuffle_inds]
        expert_actions_traj = [expert_actions_traj[i] for i in shuffle_inds]
        expert_next_states_traj = [expert_next_states_traj[i] for i in shuffle_inds]
        expert_dones_traj = [expert_dones_traj[i] for i in shuffle_inds]

        import gym 
        max_traj_length = gym.make(self.env_id)._max_episode_steps

        for idx, state in enumerate(expert_states_traj):
            states = expert_states_traj[idx]
            actions = expert_actions_traj[idx]
            next_states = expert_next_states_traj[idx]
            next_actions = expert_actions_traj[idx][1:]
            dones = expert_dones_traj[idx]
            infos = [{} for _ in range(len(states))]
            rewards = [None for _ in range(len(states))]

            # trajectory is maximum
            if len(states) == max_traj_length:
                dones = list(dones)
                dones[-1] = False
                dones = tuple(dones)
            self.transitions += list(zip(states[:-1], actions[:-1], next_states[:-1], next_actions, dones[:-1], infos[:-1]))

    def load_d4rl_data(self, filename):
        import h5py
        hfile = h5py.File(filename, 'r')
        # keys: 'actions', 'observations', 'rewards', 'timeouts', 'terminals'
        obs = hfile['observations'][:-1]
        rewards = hfile['rewards'][:-1]
        terminals = hfile['terminals'][:-1]
        timeouts = hfile['timeouts'][:-1]
        actions = hfile['actions'][:-1]
        next_obs = hfile['observations'][1:]
        next_actions = hfile['actions'][1:]

        self.transitions = list(zip(obs, actions, next_obs, next_actions, terminals, timeouts))

    def load_traj_robotics(self, filename):
        # demos generated by the fetch_data_generation.py
        #load the demonstration data from data file
        demoData = np.load(filename, allow_pickle=True) 
        obs = demoData['obs']
        acs = demoData['acs']
        info = demoData['info']
        self.trajs = [list(zip(obs[i], acs[i], info[i])) for i in range(len(obs))]
        self.traj_rets = [] # TODO

    def load_traj_mujoco(self, expert_path):
        traj_data = np.load(expert_path, allow_pickle=True)
        obs = traj_data['obs']
        acs = traj_data['acs']
        rews = traj_data['rews']
        self.trajs = [list(zip(obs[i], acs[i], rews[i])) for i in range(len(obs))]
        self.traj_rets = [sum(rews[i]) for i in range(len(obs))]
        self.plot_return_hist()
        self.make_transitions()

    def save(self, _filename):
        if self.dir_samples:
            filename = "%s/%s-%s_len_%d_return_%.2f"%(self.dir_samples, _filename, self.env_id, len(self.trajs), self.traj_rets)
        meta = {"env_id": self.env_id, "info": _filename}
        np.savez(filename, trajs=self.trajs, traj_rets=self.traj_rets, meta=meta)

    def make_transitions_with_repeated_trajs(self, num_traj_required=20):
        self.transitions = []
        self.transition_tags = []

        # traj: (states, actions, next_states, rewards, dones, infos)
        # ATTENTION: handling done signal here
        import gym 
        max_traj_length = gym.make(self.env_id)._max_episode_steps
        num_traj = 0
        while num_traj < num_traj_required:
            traj = self.trajs[num_traj % len(self.trajs)]

            states, actions, next_states, rewards, dones, infos = zip(*traj)
            # trajectory is maximum
            if len(traj) == max_traj_length:
                dones = list(dones)
                dones[-1] = False
                dones = tuple(dones)
            self.transitions += list(zip(states, actions, next_states, rewards, dones, infos))
            try: 
                self.transition_tags += [tags[num_traj]] * len(traj)
            except:
                pass

            num_traj += 1

    def make_transitions(self):
        if self.picked_trajs:
            trajs = [self.trajs[i] for i in self.picked_trajs]
            tags = [self.traj_tags[i] for i in self.pick_trajs]
        else:
            trajs = self.trajs
            tags = self.traj_tags

        self.transitions = []
        self.transition_tags = []

        # traj: (states, actions, next_states, rewards, dones, infos)
        # ATTENTION: handling done signal here
        import gym 
        max_traj_length = gym.make(self.env_id)._max_episode_steps
        for idx, traj in enumerate(trajs):
            states, actions, next_states, rewards, dones, infos = zip(*traj)
            # trajectory is maximum
            if len(traj) == max_traj_length:
                dones = list(dones)
                dones[-1] = False
                dones = tuple(dones)
            self.transitions += list(zip(states, actions, next_states, rewards, dones, infos))
            try: 
                self.transition_tags += [tags[idx]] * len(traj)
            except:
                pass

    def pick_trajs(self, picked_trajs):
        assert np.max(picked_trajs) < self.num_trajs
        self.picked_trajs = picked_trajs
        self.make_transitions()

    def collect_one_traj(self, expert_fn, max_t_per_traj=None, std=None, save_to_file=None, min_length=None, clip_action=False):
        env = make(self.env_id)
        t = 0
        is_collect = t <= max_t_per_traj if max_t_per_traj else True

        print("Start collecting samples in Env: %s"%self.env_id)
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        infos = []
        state = env.reset() # reset env
        min_length = 1 if not min_length else min_length

        while is_collect:
            action = expert_fn(state)
            # if std:
            #     action += np.random.randn(action.size) * std
            # if clip_action:
            #     action = np.clip(action, -self.act_limit, self.act_limit)
            next_s, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            next_states.append(next_s)
            rewards.append(reward)
            dones.append(done)
            info.update({'std': std})
            infos.append(info)

            if done:
                if len(states) < min_length:
                    print("Min length requirement not met")
                    states = []
                    actions = []
                    next_states = []
                    rewards = []
                    dones = []
                    infos = []
                    state = env.reset() # reset env
                    continue
                else:
                    break
            t += 1
            is_collect = t <= max_t_per_traj if max_t_per_traj else True

            state = next_s

        env.close()

        self.trajs = list(zip(states, actions, next_states, rewards, dones, infos))
        self.traj_rets = sum(rewards)
        print("Sampling over for Env: %s, returns: %.2f"%(self.env_id, sum(rewards)))

        if save_to_file:
            self.save(save_to_file)
            print("Saved samples to %s"%save_to_file)
            print("Trajectory length: %d, returns: %.2f"%(len(self.trajs), self.traj_rets))
        return self.trajs

    def sample_trajs(self, batch_size=1):
        return [self.trajs[random.randint(0, self.num_trajs)] for _ in range(batch_size)]

    def gen_sample_indices(self, batch_size, replace, mode='sa'):
        if mode == 'sa':
            num_transitions = len(self.transitions)
        else:
            raise NotImplementedError

        if batch_size < 0:
            return np.arange(num_transitions)

        if replace:
            sample_indices = [np.random.randint(0, num_transitions) for _ in range(batch_size)]
        else:
            sample_indices = np.random.choice(num_transitions, batch_size, replace=False)
        return sample_indices

    def gen_sample_indices_mixed(self, batch_size, replace):
        num_transitions = len(self.transitions + self.transitions_sampled)
        if batch_size < 0:
            return np.arange(num_transitions)
        if replace:
            sample_indices = [np.random.randint(0, num_transitions) for _ in range(batch_size)]
        else:
            sample_indices = np.random.choice(num_transitions, batch_size, replace=False)
        return sample_indices

    def get_next_batch(self, batch_size):
        return self.sample(batch_size)

    def get_all_tuples(self):
        samples = list(zip(*self.transitions))
        return samples

    def get_all_tuples_next_action(self):
        samples = list(zip(*self.transitions))
        return samples

    def sample(self, batch_size=128, replace=True):
        samples = [self.transitions[i] \
                        for i in self.gen_sample_indices(batch_size, replace)]
        samples = list(zip(*samples))
        return samples

    def sample_mixed(self, batch_size=128, replace=True):
        tmp_transitions = self.transitions + self.transitions_sampled
        samples = [tmp_transitions[i] \
                        for i in self.gen_sample_indices_mixed(batch_size, replace)]
        samples = list(zip(*samples))
        return samples

    def sample_s(self, batch_size=128, replace=True):
        samples = [self.transitions[i][0] \
                        for i in self.gen_sample_indices(batch_size, replace)]
        return samples

    def sample_sa(self, batch_size=128, replace=True):
        samples = [(self.transitions[i][0], self.transitions[i][1]) \
                        for i in self.gen_sample_indices(batch_size, replace)]
        samples = list(zip(*samples))
        return samples
    
    def sample_sasr(self, batch_size=128, replace=True):
        samples = [(self.transitions[i][0], self.transitions[i][1], self.transitions[i][2], self.transitions[i][3]) \
                        for i in self.gen_sample_indices(batch_size, replace, mode='sasr')]
        samples = list(zip(*samples))
        return samples

    def sample_random_from_env(self, batch_size=128):
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        infos = []

        for _ in range(batch_size):
            state = self.init_state
            action = self.env.action_space.sample()
            next_s, reward, done, info = self.env.step(action)

            states.append(state)
            actions.append(action)
            next_states.append(next_s)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            if done:
                self.init_state = self.env.reset() # reset env
            else: 
                self.init_state = next_s

        return states, actions, next_states, rewards

    def sample_env_init_s0(self, batch_size=128):
        env = make(self.env_id)
        samples = [env.reset() for _ in batch_size]
        return samples

    def plot_return_hist(self):
        def mkdir(path):
            from pathlib import Path
            Path(path).mkdir(parents=True, exist_ok=True)
        
        import pandas as pd
        import seaborn as sns
        sns.set(rc={'figure.figsize':(11.7,8.27)})

        data = {'returns': self.traj_rets}
        data = pd.DataFrame(data)
        plot = sns.histplot(data, x='returns', kde=True)

        mkdir('figures')
        fig = plot.get_figure()
        if self.traj_filename is None:
            self.traj_filename = self.env_id
        fig.savefig("figures/demonstration_rets_%s.png"%(self.traj_filename))

    def eval_expert(self, expert_fn, render=True, std=None, clip_action=False):
        env = make(self.env_id)
        state = env.reset()
        rewards = []
        while True:
            action = expert_fn(state)
            if std:
                action += np.random.randn(action.size) * std
            if clip_action:
                action = np.clip(action, -self.act_limit, self.act_limit)

            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
            if render:
                env.render()
        env.close()
        return rewards

    def view_traj(self, traj_idx):
        assert traj_idx < self.num_trajs
        print("Visualizing traj (%d/%d) with length %d"%(traj_idx, self.num_trajs-1, len(self.trajs[traj_idx])))
        env = make(self.env_id)
        env.reset()

        env_name = self.env_id.split('-')[0]
        if env_name in ["Hopper", "Walker2d", "HalfCheetah"]:
            x_offset = [0.0]
            offset = 1
        elif env_name in ["Ant"]:
            x_offset = [0.0, 0.0]
            offset = 2
        elif env_name in ["InvertedPendulum"]:
            x_offset = []
            offset = 0
        else:
            raise NotImplementedError

        for tran in self.trajs[traj_idx]:
            state = tran[0]
            nq = env.env.model.nq
            nv = env.env.model.nv
            qpos = np.array(x_offset + list(state[: nq-offset]))
            qvel = state[nq-offset : nq+nv-offset]
            env.env.set_state(qpos, qvel)
            env.env.sim.forward()
            env.render()
        env.close()

    def visualize_distribute(self):
        import seaborn as sns
        sns.set(rc={'figure.figsize':(11.7,8.27)})

        states, actions, tags = [], [], []
        for i in range(len(self.transitions)):
            states.append(self.transitions[i][0])
            actions.append(self.transitions[i][1])
            # extract the policy tag
            tags.append(self.transition_tags[i].split('-')[0]) 

        X = np.concatenate((states, actions), axis=-1)

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init='random', random_state=0, n_iter=1000, perplexity=30.0)
        print('Fitting to data...')
        X_embedded = tsne.fit_transform(X)
        print('Done fitting.')

        import pandas as pd 
        data = {"X": X_embedded[:, 0], "Y": X_embedded[:, 1], "policy": tags}
        data = pd.DataFrame(data)
        plot = sns.scatterplot(data=data, x="X", y="Y", hue="policy", style="policy")

        fig = plot.get_figure()
        if self.traj_filename is None:
            self.traj_filename = self.env_id
        fig.savefig("figures/sa_distribution_%s.png"%(self.traj_filename))
