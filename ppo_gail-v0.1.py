#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:49:17 2022

@author: kang
"""
# %% we first need an expert.
import gym
import seals
env_string = "seals/Walker2d-v0"
env = gym.make(env_string)

import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.ppo import MlpPolicy

generating_experts = False
if generating_experts:
    expert = SAC(policy="MlpPolicy", 
                  env = env, 
                  verbose=1,
                  tensorboard_log="/Users/kang/GitHub/GAIL-Fail/tensorboard/debug_sac_walker2dv0_expert/")
    expert.learn(1e5,tb_log_name="sac_seal_run") 
    expert.save("debug_sac_seal_expert-TVG")
else: 
    expert = SAC.load("/Users/kang/GitHub/GAIL-Fail/experts/sac_seal_expert_1.zip")

# %% We generate some expert trajectories, that the discriminator needs to distinguish from the learner's trajectories.
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(env_string))] * 32),
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

venv = DummyVecEnv([lambda: gym.make(env_string)] * 32) ## version 1

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
from imitation.scripts import train_adversarial
train_adversarial.save(gail_trainer,"/Users/kang/GitHub/GAIL-Fail/saving_ppo_gail/gail_trainer/")
learner.save("/Users/kang/GitHub/GAIL-Fail/saving_ppo_gail/gail_learner/")

# %% bc 



from imitation.algorithms import bc

transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
)

reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward before training: {reward_before_training}")

bc_trainer.train(n_epochs=int(3e8))
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 100)
print(f"Reward after training: {reward_after_training}")

from tqdm import tqdm
reward_after_training_list = []
for i in tqdm(range(100)):
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 100)
    reward_after_training_list.append(reward_after_training)
    
    
    
    
    
    
    
    
    
    
    
    