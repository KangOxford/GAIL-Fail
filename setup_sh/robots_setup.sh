git config --global user.email "kang.li@maths.ox.ac.uk"
git config --global user.name "KangOxford"

wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -zxvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco 
rm mujoco210-linux-x86_64.tar.gz

wget http://www.roboti.us/file/mjkey.txt
cp mjkey.txt ~/.mujoco/mjkey.txt
rm mjkey.txt



sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
sudo apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common patchelf

wget "https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh"
bash Anaconda3-2019.03-Linux-x86_64.sh
rm Anaconda3-2019.03-Linux-x86_64.sh
export PATH=~/anaconda3/bin:$PATH
conda init bash


echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kang/.mujoco/mujoco210/bin' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc 
source ~/.bashrc 

pip install seals
pip install stable-baselines3
pip install tensorboard
pip install mujoco-py
pip install imitation
pip install tensorflow



git clone https://github.com/tadashiK/mujoco-py
cp ~/.mujoco/mjkey.txt mujoco-py/singularity
cd mujoco-py/singularity

