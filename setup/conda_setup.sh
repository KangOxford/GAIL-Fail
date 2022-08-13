wget "https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh"

bash Anaconda3-2019.03-Linux-x86_64.sh

rm Anaconda3-2019.03-Linux-x86_64.sh

export PATH=/workspace/anaconda3/bin:$PATH
/workspace/anaconda3/bin/conda init init bash 
conda create -n gail python=3.7.3
