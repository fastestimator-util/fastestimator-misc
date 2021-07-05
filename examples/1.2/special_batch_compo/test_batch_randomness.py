import pdb

import numpy as np
import tensorflow as tf
from numpy.core.fromnumeric import trace
from torch.utils.data import Dataset

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset import BatchDataset, NumpyDataset
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp


def create_dataset1():
    images = np.random.rand(5, 28, 28)
    labels = [0, 1, 2, 3, 4]
    return NumpyDataset({"x": images, "y": labels})


def create_dataset2():
    images = np.random.rand(5, 28, 28)
    labels = [5, 6, 7, 8, 9]
    return NumpyDataset({"x": images, "y": labels})


class DebugTrace(fe.trace.Trace):
    def on_batch_end(self, data):
        print("==new batch====")
        print(data["y"].numpy())


def get_estimator():
    ds1 = create_dataset1()
    ds2 = create_dataset2()
    batch_ds = BatchDataset(datasets=(ds1, ds2), num_samples=(1, 1))
    pipeline = fe.Pipeline(train_data=batch_ds,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=4, traces=DebugTrace(inputs="y"))
    return estimator
