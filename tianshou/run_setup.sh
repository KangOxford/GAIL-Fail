sudo apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common
sudo apt-get install -y patchelf

pip install --upgrade pip
pip install free-mujoco-py

# pip install opencv-python
# pip install PyHamcrest
# pip install protobuf
# pip install tensorflow
# pip install gym
# pip install numpy
# pip install pyyaml
# pip install termcolor
# pip install json_tricks
# pip install ipython
pip install tianshou

pip install gym[classic_control]

git config --global user.email "kang.li@maths.ox.ac.uk"
git config --global user.name "KangOxford"


# cd /workspace/.pyenv_mirror/user/3.8.13/lib/python3.8/site-packages/tianshou





