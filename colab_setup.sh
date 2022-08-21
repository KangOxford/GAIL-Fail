# sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
# sudo apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libgl

# mkdir ~/.mujoco 
# wget http://www.roboti.us/download/mjpro150_osx.zip
# unzip mjpro150_osx.zip -d ~/.mujoco 
# rm mjpro150_osx.zip

# wget http://www.roboti.us/download/mujoco200_macos.zip
# unzip mujoco200_macos.zip -d ~/.mujoco 
# rm mujoco200_macos.zip

# wget http://www.roboti.us/file/mjkey.txt
# cp mjkey.txt ~/.mujoco/mjkey.txt
# rm mjkey.txt

# export PATH=~/.mujoco/mjpro150/bin:$PATH
# export PATH=~/.mujoco/mjpro200_linux/bin:$PATH
# export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mjpro150/bin:$MUJOCO_PY_MUJOCO_PATH
# export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mjpro200_linux/bin:$MUJOCO_PY_MUJOCO_PATH

pip install seals
pip install stable-baselines3
pip install tensorboard
pip install mujoco_py
pip install imitation