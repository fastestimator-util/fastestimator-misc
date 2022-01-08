#!/bin/bash

# for open cv
apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender1 libxext6

# backends
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# FE
pip install fastestimator

# training script, if you have specific training data, make use the $SM_CHANNEL_TRAINING
fastestimator run mnist_tf.py --save_dir $SM_MODEL_DIR  --restore_dir /opt/ml/checkpoints/
