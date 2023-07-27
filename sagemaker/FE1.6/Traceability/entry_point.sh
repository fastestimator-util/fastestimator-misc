#!/bin/bash

chmod 1777 /tmp
apt-get update --allow-unauthenticated

# for open cv
apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender1 libxext6

# for installation of latex for traceability
unset PYTHONPATH
export DEBIAN_FRONTEND=noninteractive
apt-get install -y graphviz texlive-latex-base texlive-latex-extra

# backends
pip install tensorflow==2.11.1
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# FE
pip install fastestimator==1.6.0

# training command, notice the experiment name is needed for traceability by --summary. If you have specific training data, make use the $SM_CHANNEL_TRAINING
fastestimator train mnist_tf.py --save_dir $SM_MODEL_DIR --summary tf_mnist