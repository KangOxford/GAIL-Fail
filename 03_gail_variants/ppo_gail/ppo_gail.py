#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:49:17 2022

@author: kang

just used for testing 
do not run experiment based on it
"""
# %% we first need an expert.

from gc import callbacks
import seals
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import gym
import numpy as np
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

env_string = "seals/Walker2d-v0"
env = gym.make(env_string)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
expert = SAC.load("/home/kang/GAIL-Fail/02_generated_experts/expert_sac_robots_cuda-v8.zip")
print(">>> Load pretrained experts")

#  We generate some expert trajectories, that the discriminator needs to distinguish from the learner's trajectories.
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    make_vec_env(
        env_string,
        n_envs=5,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    ),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)

#  Now we are ready to set up our GAIL trainer.
# Note, that the `reward_net` is actually the network of the discriminator.
# We evaluate the learner before and after training so we can see if it made any progress.
# firstly build the env
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import gym


venv = make_vec_env(env_string, n_envs=1)

#  define the learner
from stable_baselines3.ppo import MlpPolicy
learner = PPO(
    policy=MlpPolicy,
    env=venv,
    verbose=1,
    )
reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)
#  costum logger 

from stable_baselines3.common.logger import configure
tmp_path = "/home/kang/GAIL-Fail/long_run_ppo_gail/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


from imitation.util.logger import HierarchicalLogger
new_logger = HierarchicalLogger(new_logger, ["stdout", "log","csv"])

#  define the GAIL
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    custom_logger = new_logger,
    log_dir = tmp_path,
)
#  training the GAIL 1/2
learner_rewards_before_training, _ = evaluate_policy(
    learner, venv, 10, return_episode_rewards=True
)
print(">>> learner_rewards_before_training",learner_rewards_before_training)
print(">>> np.mean(learner_rewards_before_training) ",np.mean(learner_rewards_before_training))
#  training the GAIL 2/2

for i in range(int(1e6)):
    print("="*20 + 'Epoch'+str(i)+' '+'='*20)
    gail_trainer.train(int(3e5))  # Note: set to 300000 for better results
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 10, return_episode_rewards=True
    )
    print(">>> learner_rewards_after_training",learner_rewards_after_training)
    print(">>> np.mean(learner_rewards_after_training) ",np.mean(learner_rewards_after_training))
