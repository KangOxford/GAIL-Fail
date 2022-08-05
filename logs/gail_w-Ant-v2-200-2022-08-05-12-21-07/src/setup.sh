

sudo apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common

sudo apt-get install -y patchelf

pip install opencv-python
pip install PyHamcrest==1.9.0
pip install protobuf==3.20.*
pip install tensorflow==1.13.1
pip install gym==0.15.6
pip install free-mujoco-py
pip install numpy
pip install pyyaml
pip install termcolor
pip install json_tricks

git config --global user.email "kang.li@maths.ox.ac.uk"
git config --global user.name "KangOxford"

# bash ./scripts/run_gail.sh





