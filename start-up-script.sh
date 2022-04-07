# #!/bin/bash

# Install Anconda
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash ./Anaconda3-5.0.1-Linux-x86_64.sh -b -p anaconda3
export PATH=/anaconda3/bin:$PATH

echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  apt-get update
  apt-get install cuda-9-0 -y
fi

# Run setup
git clone https://github.com/rohanchandra30/Spectral-Trajectory-Prediction.git
cd Spectral-Trajectory-Prediction
conda env create -f env.yml
source activate sc-glstm
python setup.py
