#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:49:17 2022

@author: kang

This version is the ppo_gail using cpu and expert learns 1e8 epoches
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
from stable_baselines3.common.vec_env import DummyVecEnv

env_string = "seals/Walker2d-v0"
num_cpu = 8
env = Monitor(gym.make(env_string))
# venv = DummyVecEnv([lambda: gym.make(env_string)] * num_cpu)
expert = SAC(policy="MlpPolicy", 
                env = env, 
                verbose=1,
                tensorboard_log="/home/kang/GAIL-Fail/tensorboard/expert_sac_robots-v6.2-trained-v2/",
                device = "cuda"
                )
expert.learn(int(1e3),tb_log_name="sac_robots_cuda_run")
for i in range(int(1e2)):
    expert.learn(int(1e6),tb_log_name="sac_robots_cuda_run", reset_num_timesteps=False) 
    expert.save("home/kang/GAIL-Fail/experts/linux_generated/expert_sac_robots_cuda-v6.2-trained-v2.zip")
    # import time;file_string = "/home/kang/GAIL-Fail/experts/linux_generated/"+str(int(time.time()))
    # expert.save(file_string + "/expert_sac_robots_cpu-v6.2-train.zip")
