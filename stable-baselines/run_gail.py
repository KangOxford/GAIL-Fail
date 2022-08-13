import gym
from stable_baselines import GAIL, SAC
from stable_baselines.gail import ExpertDataset, generate_expert_traj

# Generate expert trajectories (train expert)
# model = SAC('MlpPolicy', 'Walker2d-v2', verbose=1)
model = SAC('MlpPolicy', 'Walker2d-v2')
generate_expert_traj(model, 'expert_walker2d', n_timesteps=100, n_episodes=10)

# Load the expert dataset
# dataset = ExpertDataset(expert_path='expert_walker2d.npz', traj_limitation=10, verbose=1)
dataset = ExpertDataset(expert_path='expert_walker2d.npz', traj_limitation=10)

model = GAIL('MlpPolicy', 'Walker2d-v2', dataset)
# model = GAIL('MlpPolicy', 'Walker2d-v2', dataset, verbose=1)
# Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=1000)
model.save("gail_walker2d")

# del model # remove to demonstrate saving and loading

# model = GAIL.load("gail_pendulum")

# env = gym.make('Pendulum-v0')
obs = env.reset()
import time 
while True:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
  print(f"rewards:{rewards}")
  print(f"obs: {obs}")
  print(f"done: {dones}")
  time.sleep(3)
#   env.render()