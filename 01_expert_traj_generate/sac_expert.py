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
from stable_baselines3.common.monitor import Monitor

env_string = "seals/Walker2d-v0"
env = Monitor(gym.make(env_string))
expert = SAC(policy="MlpPolicy", 
                env = env, 
                verbose=1,
                tensorboard_log="/home/scat9001/GAIL-Fail/tensorboard/expert_sac_arc/",
                device = "cpu")
expert.learn(1e8,tb_log_name="sac_arc") 
expert.save("a","/home/scat9001/GAIL-Fail/02_generated_experts/expert_sac_arc")