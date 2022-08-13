# sudo apt update && sudo apt install -y --allow-unauthenticated wget build-essential libosmesa6-dev libglew-dev
# sudo apt install -y python3-dev patchelf
# sudo apt install -y openmpi-bin openmpi-common openssh-client libopenmpi-dev zlib1g-dev unzip

mkdir -p /workspace/.mujoco 
wget http://www.roboti.us/download/mjpro150_linux.zip
unzip mjpro150_linux.zip -d /workspace/.mujoco 
rm mjpro150_linux.zip

wget http://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip -d /workspace/.mujoco 
rm mujoco200_linux.zip

wget http://www.roboti.us/file/mjkey.txt
cp mjkey.txt /workspace/.mujoco/mjkey.txt
rm mjkey.txt

# export LD_LIBRARY_PATH=/home/$USER/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/$USER/.mujoco/mjpro200_linux/bin:$LD_LIBRARY_PATH

export PATH=/workspace/.mujoco/mjpro150/bin:$PATH
export PATH=/workspace/.mujoco/mjpro200_linux/bin:$PATH








