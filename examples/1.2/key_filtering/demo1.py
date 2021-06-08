import tempfile
from os import pipe

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


def create_pipeline():
    train_data, eval_data = mnist.load_data()
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           batch_size=32,
                           ops=[ExpandDims(inputs="x", outputs="x1"), Minmax(inputs="x1", outputs="x")])
    return pipeline


def get_estimator():
    # step 1
    pipeline = create_pipeline()
    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=2)
    return estimator



if __name__ == "__main__":
    pipeline = create_pipeline()
    data = pipeline.get_results()
    print(data.keys())
