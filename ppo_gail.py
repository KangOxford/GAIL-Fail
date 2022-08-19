#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:49:17 2022

@author: kang
"""
# %% we first need an expert.
import gym
import seals
env = gym.make("seals/Walker2d-v0")

import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
expert = DDPG(
    policy="MlpPolicy",
    env=env,
    action_noise=action_noise, 
    verbose=1,
    tensorboard_log="/Users/kang/GitHub/GAIL-Fail/tensorboard/sac_walker2dv2_expert/"
)
# expert = PPO(
#     policy=MlpPolicy,
#     env=env,
#     seed=0,
#     verbose=1,
#     tensorboard_log="/Users/kang/GitHub/GAIL-Fail/tensorboard/sac_walker2dv2_expert/"
# )
# expert = SAC(policy="MlpPolicy", 
#              env = env, 
#              verbose=1,
#              tensorboard_log="/Users/kang/GitHub/GAIL-Fail/tensorboard/sac_walker2dv2_expert/")
expert.learn(1e20,tb_log_name="ddpg_seal_run") 

# %% We generate some expert trajectories, that the discriminator needs to distinguish from the learner's trajectories.

