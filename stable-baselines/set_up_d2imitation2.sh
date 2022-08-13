export PATH=/home/$USER/.mujoco/mjpro150/bin:$PATH
export PATH=/home/$USER/.mujoco/mjpro200_linux/bin:$PATH

pip install --upgrade pip
pip install cffi
pip install mujoco-py==1.50.1.68
pip install gym==0.15.4
pip install seaborn==0.11.0
pip install pyquaternion==0.9.5
pip install joblib==0.13.2
pip install scikit-learn==0.21.3
pip install mpi4py==3.0.1
pip install tqdm==4.55.0
pip install box2d==2.3.10