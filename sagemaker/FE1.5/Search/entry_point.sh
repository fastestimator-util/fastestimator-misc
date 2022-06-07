#!/bin/bash

# for open cv
apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender1 libxext6

# backends
pip install tensorflow==2.9.1
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# FE
pip install fastestimator

# training script, if you have specific training data, make use the $SM_CHANNEL_TRAINING
fastestimator run mnist_tf.py --save_dir $SM_MODEL_DIR  --restore_dir /opt/ml/checkpoints/
