export PATH=/workspace/.mujoco/mjpro150/bin:$PATH
export PATH=/workspace/.mujoco/mjpro200_linux/bin:$PATH
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mjpro150/bin:$MUJOCO_PY_MUJOCO_PATH
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mjpro200_linux/bin:$MUJOCO_PY_MUJOCO_PATH
export PATH=/workspace/anaconda3/bin:$PATH

# export PATH=/workspace/GAIL-Fail/stable-baselines/d2-imitation:$PATH
# export PATH=/workspace/GAIL-Fail/stable-baselines/d2-imitation/baselines:$PATH


sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
sudo apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf

conda init bash
conda activate gail
ipython


# import os
# os.sys.path.append("/workspace/GAIL-Fail/stable-baselines/d2-imitation")

# pip install hrl-pybullet-envs ## not sure about whether it can make 'import pybullet_envs' work


