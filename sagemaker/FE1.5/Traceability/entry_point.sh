#!/bin/bash

# update package links
chmod 1777 /tmp
apt update --allow-unauthenticated

# for opencv
apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender1 libxext6 git

# for installation of latex for traceability
unset PYTHONPATH
export DEBIAN_FRONTEND=noninteractive
apt-get install -y graphviz texlive-latex-base texlive-latex-extra

# backends
pip install tensorflow==2.9.1
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# FE
pip install fastestimator

# training command, notice the experiment name is needed for traceability by --summary. If you have specific training data, make use the $SM_CHANNEL_TRAINING
fastestimator train mnist_tf.py --save_dir $SM_MODEL_DIR --summary tf_mnist
