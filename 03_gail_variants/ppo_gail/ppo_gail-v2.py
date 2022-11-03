import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy 
import seals # needed to load environments
env = gym.make("seals/Walker2d-v0")
# expert = PPO(   
#     policy=MlpPolicy,
#     env=env,
#     seed=0,
#     batch_size=64,
#     ent_coef=0.0,
#     learning_rate=0.0003,
#     n_epochs=10,
#     n_steps=64,
# )
# expert.learn(1000) # Note: set to 100000 to train a proficient expert
expert = PPO.load("/home/kang/GAIL-Fail/02_generated_experts/expert_sac_robots_cuda-v8.zip")
print(">>> Load pretrained experts")

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper 
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv 
import numpy as np
rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    make_vec_env(
    "seals/Walker2d-v0",
n_envs=5,
post_wrappers=[lambda env, _: RolloutInfoWrapper(env)], rng=rng,
),
rollout.make_sample_until(min_timesteps=None, min_episodes=60), rng=rng,
)



from imitation.algorithms.adversarial.gail import GAIL 
from imitation.rewards.reward_nets import BasicRewardNet 
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
venv = make_vec_env("seals/Walker2d-v0", n_envs=8, rng=rng)
learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
)
reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)
# learner_rewards_before_training, _ = evaluate_policy( learner, venv, 100, return_episode_rewards=True
# )
gail_trainer.train(20000) # Note: set to 300000 for better results learner_rewards_after_training, _ = evaluate_policy(
learner_rewards_after_training, _ = evaluate_policy(
learner, venv, 100, return_episode_rewards=True )
