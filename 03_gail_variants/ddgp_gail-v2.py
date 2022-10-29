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

# %% We generate some expert trajectories, that the discriminator needs to distinguish from the learner's trajectories.
rng = np.random.default_rng(0)
print("Sampling expert transitions.")
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)



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

venv = DummyVecEnv([lambda: gym.make(env_string)] * 1) ## version 3
# venv = gym.make(env_string) ## version 2
# venv = DummyVecEnv([lambda: gym.make(env_string)] * 8) ## version 1

# %% define the learner
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
tmp_path = "/home/kang/GAIL-Fail/tb_ddpg_gail_stats/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


from imitation.util.logger import HierarchicalLogger
new_logger = HierarchicalLogger(new_logger, ["stdout", "log","csv"])

# %% define the GAIL
gail_trainer = GAIL(
    demonstrations=transitions,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    custom_logger = new_logger,
    log_dir = tmp_path,
    callbacks = evaluate_policy(
    learner, venv, 10, return_episode_rewards=True
)
)

# %% traning the GAIL 1/2
# learner_rewards_before_training, _ = evaluate_policy(
#     learner, venv, 10, return_episode_rewards=True
# )

# learner_rewards_before_training, _ = evaluate_policy(
#     learner, venv, 100, return_episode_rewards=True
# )

# %% training the GAIL 2/2
gail_trainer.train(int(1e2))  # Note: set to 300000 for better results

learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)

# %% When we look at the histograms of rewards before and after learning, 
# we can see that the learner is not perfect yet, but it made some progress at least.
# If not, just re-run the above cell.
import matplotlib.pyplot as plt
import numpy as np

# print("np.mean(learner_rewards_before_training) ",np.mean(learner_rewards_before_training))
print("np.mean(learner_rewards_after_training) ",np.mean(learner_rewards_after_training))

plt.hist(
    [learner_rewards_before_training, learner_rewards_after_training],
    label=["untrained", "trained"],
)
plt.legend()
plt.show()