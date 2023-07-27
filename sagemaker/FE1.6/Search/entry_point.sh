#!/bin/bash

# for open cv
apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender1 libxext6

# backends
pip install tensorflow==2.11.1
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# FE
pip install fastestimator==1.6.0

# training script, if you have specific training data, make use the $SM_CHANNEL_TRAINING
fastestimator run mnist_tf.py --save_dir $SM_MODEL_DIR  --restore_dir /opt/ml/checkpoints/