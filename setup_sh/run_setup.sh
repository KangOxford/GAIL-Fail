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
elif ["$uname"=="Colab"];then
    export PATH=/workspace/.mujoco/mjpro150/bin:$PATH
    export PATH=/workspace/.mujoco/mjpro200_linux/bin:$PATH
    export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mjpro150/bin:$MUJOCO_PY_MUJOCO_PATH
    export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mjpro200_linux/bin:$MUJOCO_PY_MUJOCO_PATH
    # export PATH=/workspace/anaconda3/bin:$PATH
    # export PATH=/workspace/GAIL-Fail/stable-baselines/d2-imitation:$PATH
    # export PATH=/workspace/GAIL-Fail/stable-baselines/d2-imitation/baselines:$PATH
    sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
    sudo apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common patchelf

    # conda init bash
    # conda activate gail
    # ipython

    # import os
    # os.sys.path.append("/workspace/GAIL-Fail/stable-baselines/d2-imitation")
    # pip install hrl-pybullet-envs ## not sure about whether it can make 'import pybullet_envs' work
    # git rm -r --cached .

# ============================= run on linux =============================
elif ["$(uname)"=="Linux"];then
    wget "https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh"
    bash Anaconda3-2019.03-Linux-x86_64.sh
    rm Anaconda3-2019.03-Linux-x86_64.sh
    export PATH=~/anaconda3/bin:$PATH
    conda init bash

    sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
    sudo apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common patchelf

    mkdir /home/$USER/.mujoco 
    wget http://www.roboti.us/download/mjpro150_linux.zip
    unzip mjpro150_linux.zip -d /home/$USER/.mujoco 
    rm mjpro150_linux.zip

    wget http://www.roboti.us/download/mujoco200_linux.zip
    unzip mujoco200_linux.zip -d /home/$USER/.mujoco 
    rm mujoco200_linux.zip

    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
    tar -zxvf mujoco210-linux-x86_64.tar.gz -C /home/$USER/.mujoco 
    rm mujoco210-linux-x86_64.tar.gz

    wget http://www.roboti.us/file/mjkey.txt
    cp mjkey.txt /home/$USER/.mujoco/mjkey.txt
    rm mjkey.txt

    export PATH=/home/$USER/.mujoco/mjpro150/bin:$PATH
    export PATH=/home/$USER/.mujoco/mujoco200_linux/bin:$PATH
    export MUJOCO_PY_MUJOCO_PATH=/home/$USER/.mujoco/mjpro150/bin:$MUJOCO_PY_MUJOCO_PATH
    export MUJOCO_PY_MUJOCO_PATH=/home/$USER/.mujoco/mujoco200_linux/bin:$MUJOCO_PY_MUJOCO_PATH
    
    export PATH=~/.mujoco/mujoco210/bin:$PATH
    export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210/bin:$MUJOCO_PY_MUJOCO_PATH
    export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH

    export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH

    # sudo apt install python-pip
    pip install seals
    pip install stable-baselines3
    pip install tensorboard
    pip install mujoco_py
    pip install imitation
    
# ============================= run when debug =============================
elif ["$(uname)"=="Debug"];then
    git clone https://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@github.com/KangOxford/GAIL-Fail.git
    git config --global user.email "kang.li@maths.ox.ac.uk"
    git config --global user.name "KangOxford"
    tensorboard --logdir /home/kangli/GAIL-Fail/tensorboard/debug_sac_walker2dv0_expert/
fi
