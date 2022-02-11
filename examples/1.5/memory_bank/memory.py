import pdb
import random
import tempfile

import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop import LambdaOp, TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace import Trace


class MemoryBank(Trace):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.memory_bank = {}
        self.feature_vector_shape = (64,)

    def on_batch_begin(self, data):
        labels = data["y"].numpy()
        feature_selected = []
        for label in labels:
            if label not in self.memory_bank:
                feature_selected.append(np.zeros(self.feature_vector_shape))
            else:
                feature_selected.append(random.choice(self.memory_bank[label]))
        feature_selected = tf.convert_to_tensor(np.array(feature_selected, dtype="float32"))
        data.maps[1]["feature_selected"] = feature_selected

    def on_batch_end(self, data):
        feature_vectors = data["feature_vector"].numpy()
        labels = data["y"].numpy()
        for label, feature_vector in zip(labels, feature_vectors):
            label = int(label)
            if label not in self.memory_bank:
                self.memory_bank[label] = []
            self.memory_bank[label].append(feature_vector)

class CustomLoss(TensorOp):
    def forward(self, data, state):
        feature_vector, feature_selected = data
        if tf.reduce_sum(feature_selected) == 0:
            loss = 0.0
        else:
            loss = tf.reduce_sum(tf.abs(feature_selected - feature_vector))
        return loss

def get_estimator(epochs=2, batch_size=32):
    # step 1
    train_data, eval_data = mnist.load_data()
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           batch_size=batch_size,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])
    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs=["y_pred", "feature_vector"], intermediate_layers='dense'),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        CustomLoss(inputs=("feature_vector", "feature_selected"), outputs="feature_loss"),
        LambdaOp(fn=lambda x, y: x +y, inputs=("ce", "feature_loss"), outputs="total_loss"),
        UpdateOp(model=model, loss_name="total_loss")
    ])
    # step 3
    traces = [MemoryBank(inputs=("feature_vector", "y"), outputs="feature_selected")]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces)
    return estimator

est = get_estimator()
est.fit(warmup=False)