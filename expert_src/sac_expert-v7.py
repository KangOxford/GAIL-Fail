#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:49:17 2022

@author: kang

This version is the ppo_gail using gpu and expert learns 1e8 epoches
and env = env with 1 times.
can show the rollout results
"""
# %% we first need an expert.
import gym
import seals
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.ppo import MlpPolicy

env_string = "seals/Walker2d-v0"
generating_experts = False
if generating_experts:
    env = gym.make(env_string)
    expert = SAC(policy="MlpPolicy", 
                  env = env, 
                  verbose=1,
                  tensorboard_log="/home/kang/GAIL-Fail/tensorboard/expert_sac_robots/",
                  device = "cuda")
    expert.learn(1e8,tb_log_name="sac_robots_run") 
    expert.save("a","expert_sac_robots_v6")
else: 
    expert = SAC.load("/home/kang/GAIL-Fail/experts/sac_seal_expert_1.zip")

# %% We generate some expert trajectories, that the discriminator needs to distinguish from the learner's trajectories.
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(env_string))] * 64),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
)
# %% Now we are ready to set up our GAIL trainer.
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

venv = DummyVecEnv([lambda: gym.make(env_string)] * 64) ## version 1

# %% define the learner
from stable_baselines3.ppo import MlpPolicy
learner = PPO(
    policy=MlpPolicy,
    env=venv,
    verbose=1,
    tensorboard_log=\
        "/Users/kang/GitHub/GAIL-Fail/tensorboard/debug_mimic_ppo_gail/"
    )
reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)

# %% define the GAIL
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

# %% traning the GAIL 1/2
learner_rewards_before_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)
# %% training the GAIL 2/2
# gail_trainer.train(int(1e5))  # Note: set to 300000 for better results
gail_trainer.train(int(3e8))  # Note: set to 300000 for better results
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
# %%
