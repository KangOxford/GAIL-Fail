#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:49:17 2022

@author: kang

just used for testing 
do not run experiment based on it
"""
# %% we first need an expert.
import gym
import seals
env_string = "seals/Walker2d-v0"
env = gym.make(env_string)
testing = True

import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

expert = SAC.load("/home/kang/GAIL-Fail/experts/linux_generated/1662757477/expert_sac_robots_cuda-v8.zip")
print(">>> Load pretrained experts")

#  We generate some expert trajectories, that the discriminator needs to distinguish from the learner's trajectories.
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(env_string))] * 5),
    rollout.make_sample_until(min_timesteps=None, min_episodes=100),
)

#  Now we are ready to set up our GAIL trainer.
# Note, that the `reward_net` is actually the network of the discriminator.
# We evaluate the learner before and after training so we can see if it made any progress.
# firstly build the env
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import gym
import seals

venv = DummyVecEnv([lambda: gym.make(env_string)] * 1) ## version 3
# venv = gym.make(env_string) ## version 2
# venv = DummyVecEnv([lambda: gym.make(env_string)] * 8) ## version 1

#  define the learner
learner = DDPG(
    policy="MlpPolicy",
    env=venv,
    action_noise=action_noise, 
)
reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)
# %% costum logger 

from stable_baselines3.common.logger import configure
tmp_path = "/home/kang/GAIL-Fail/train_tb_ddpg_gail_3.5/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


from imitation.util.logger import HierarchicalLogger
new_logger = HierarchicalLogger(new_logger, ["stdout", "log","csv"])

# %% define the GAIL
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    custom_logger = new_logger,
    log_dir = tmp_path
)

# %% traning the GAIL 1/2
learner_rewards_before_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)
# %% training the GAIL 2/2
gail_trainer.train(int(10e6))  # Note: set to 300000 for better results

learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)
# %% When we look at the histograms of rewards before and after learning, 
# we can see that the learner is not perfect yet, but it made some progress at least.
# If not, just re-run the above cell.
import matplotlib.pyplot as plt
import numpy as np

print("np.mean(learner_rewards_after_training) ",np.mean(learner_rewards_after_training))
print("np.mean(learner_rewards_before_training) ",np.mean(learner_rewards_before_training))

plt.hist(
    [learner_rewards_before_training, learner_rewards_after_training],
    label=["untrained", "trained"],
)
plt.legend()
plt.show()