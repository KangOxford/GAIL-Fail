import pickle
import os
import time
import yaml
import random
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger, log_kvs


#TODO change this part
from trpo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from trpo.v_function.mlp_v_function import MLPVFunction
from trpo.algos.trpo import TRPO
from trpo.utils.normalizer import Normalizers
#TODO change this part

from gail.discriminator.discriminator import Discriminator
from gail.discriminator.linear_reward import LinearReward
# (TimeStep, ReplayBuffer) are required to restore from pickle.
from gail.utils.replay_buffer import TimeStep, ReplayBuffer, load_expert_dataset
from gail.utils.runner import Runner, evaluate
from utils import FLAGS, get_tf_config


ENV="Walker2d-v2"
NUM_ENV=1
SEED=200
BUF_LOAD="/content/drive/MyDrive/GitHub/GAIL-Fail/project_2022_05_06/dataset/sac/"+ENV
VF_HIDDEN_SIZES=100
D_HIDDEN_SIZES=100
POLICY_HIDDEN_SIZES=100
# Discriminator
NEURAL_DISTANCE=True
GRADIENT_PENALTY_COEF=10.0
L2_REGULARIZATION_COEF=0.0
REWARD_TYPE="nn"
# Learning
TRPO_ENT_COEF=0.0
LEARNING_ABSORBING=False
TRAJ_LIMIT=3
TRAJ_SIZE=1000
ROLLOUT_SAMPLES=1000
TOTAL_TIMESTEPS=3000000


FLAGS.seed=SEED 
FLAGS.algorithm="gail_w" 
FLAGS.env.id=ENV 
FLAGS.env.num_env=NUM_ENV 
FLAGS.env.env_type="mujoco" 
FLAGS.GAIL.buf_load=BUF_LOAD 
FLAGS.GAIL.learn_absorbing=LEARNING_ABSORBING 
FLAGS.GAIL.traj_limit=TRAJ_LIMIT 
FLAGS.GAIL.trajectory_size=TRAJ_SIZE 
FLAGS.GAIL.reward_type=REWARD_TYPE 
FLAGS.GAIL.discriminator.neural_distance=NEURAL_DISTANCE 
FLAGS.GAIL.discriminator.hidden_sizes=D_HIDDEN_SIZES 
FLAGS.GAIL.discriminator.gradient_penalty_coef=GRADIENT_PENALTY_COEF 
FLAGS.GAIL.discriminator.l2_regularization_coef=L2_REGULARIZATION_COEF 
FLAGS.GAIL.total_timesteps=TOTAL_TIMESTEPS 
FLAGS.TRPO.rollout_samples=ROLLOUT_SAMPLES 
FLAGS.TRPO.vf_hidden_sizes=VF_HIDDEN_SIZES 
FLAGS.TRPO.policy_hidden_sizes=POLICY_HIDDEN_SIZES 
FLAGS.TRPO.algo.ent_coef=TRPO_ENT_COEF

from gail.main import *
with tf.Session(config=get_tf_config()):
    main()

# for ENV in ("Walker2d-v2","HalfCheetah-v2","Hopper-v2"):
#     for SEED in (100,200,300):
        # main()
        