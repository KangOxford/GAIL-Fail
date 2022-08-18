# =============================  run on mac  =============================
if ["$(uname)"=="Darwin"];then
    brew install llvm boost hdf5
    brew install openmpi
    brew install gcc
    brew install wget 
    # brew install build-essential 
    # brew install libosmesa6-dev 
    # brew install libglew-dev

    mkdir /Users/$USER/.mujoco 
    wget http://www.roboti.us/download/mjpro150_osx.zip
    unzip mjpro150_osx.zip -d /Users/$USER/.mujoco 
    rm mjpro150_osx.zip

    wget http://www.roboti.us/download/mujoco200_macos.zip
    unzip mujoco200_macos.zip -d /Users/$USER/.mujoco 
    rm mujoco200_macos.zip

    wget http://www.roboti.us/file/mjkey.txt
    cp mjkey.txt /Users/$USER/.mujoco/mjkey.txt
    rm mjkey.txt

    export PATH=/Users/$USER/.mujoco/mjpro150/bin:$PATH
    export PATH=/Users/$USER/.mujoco/mujoco200_macos/bin:$PATH
    export MUJOCO_PY_MUJOCO_PATH=/Users/$USER/.mujoco/mjpro150/bin:$MUJOCO_PY_MUJOCO_PATH
    export MUJOCO_PY_MUJOCO_PATH=/Users/$USER/.mujoco/mujoco200_macos/bin:$MUJOCO_PY_MUJOCO_PATH


# ============================= run on colab =============================
elif ["$uname"=="Linux"];then
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

    # conda init bash
    # conda activate gail
    # ipython

    # import os
    # os.sys.path.append("/workspace/GAIL-Fail/stable-baselines/d2-imitation")
    # pip install hrl-pybullet-envs ## not sure about whether it can make 'import pybullet_envs' work
    # git rm -r --cached .
fi