import pdb

import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.trace import Trace


class DataCache(Trace):
    def on_epoch_begin(self, data):
        self.cached_input = None
        self.cached_label = None

    def on_batch_end(self, data):
        self.cached_input = data["x"][:16]
        self.cached_label = data["y"][:16]

    def on_batch_begin(self, data):
        if self.cached_input is None:
            self.cached_input = tf.convert_to_tensor(np.zeros((16, 28, 28, 1), dtype="float32"))
            self.cached_label = tf.convert_to_tensor(np.zeros((16, ), dtype="uint8"))
        data.write_without_log("x_cache", self.cached_input)
        data.write_without_log("y_cache", self.cached_label)


class CombineCache(TensorOp):
    def forward(self, data, state):
        x, x_cache, y, y_cache = data
        if tf.reduce_sum(x_cache) > 0:
            x_combined = tf.concat([x, x_cache], axis=0)
            y_combined = tf.concat([y, y_cache], axis=0)
        else:
            x_combined = tf.concat([x, x[:16]], axis=0)
            y_combined = tf.concat([y, y[:16]], axis=0)
        return x_combined, y_combined


def get_estimator(epochs=2, batch_size=32):
    train_data, _ = mnist.load_data()
    pipeline = fe.Pipeline(train_data=train_data,
                           batch_size=batch_size,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    network = fe.Network(ops=[
        CombineCache(inputs=("x", "x_cache", "y", "y_cache"), outputs=("x_combined", "y_combined"), mode="train"),
        ModelOp(model=model, inputs="x_combined", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y_combined"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=DataCache(inputs=("x", "y"), outputs=("x_cache", "y_cache"), mode="train"))
    return estimator
