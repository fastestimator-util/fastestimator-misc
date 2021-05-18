#!/bin/bash

# for open cv
apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender1 libxext6

# backends
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorflow==2.5.0

# FE
pip install fastestimator

# training script, if you have specific training data, make use the $SM_CHANNEL_TRAINING
fastestimator train mnist_tf.py --save_dir $SM_MODEL_DIR
