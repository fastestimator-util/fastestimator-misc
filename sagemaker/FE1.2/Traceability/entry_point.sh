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
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorflow==2.5.0

# FE
pip install fastestimator

# training command, notice the experiment name is needed for traceability by --summary. If you have specific training data, make use the $SM_CHANNEL_TRAINING
fastestimator train mnist_tf.py --save_dir $SM_MODEL_DIR --summary tf_mnist
