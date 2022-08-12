sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
sudo apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf

pip install --upgrade pip
pip install gym
pip install free-mujoco-py
pip install mpi4py
pip install stable-baselines[mpi]