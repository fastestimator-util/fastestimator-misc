import tempfile

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


def get_estimator(epochs=2, batch_size=32, save_dir=tempfile.mkdtemp()):
    # step 1
    train_data, eval_data = mnist.load_data()
    test_data = eval_data.split(0.5)
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           test_data=test_data,
                           batch_size=batch_size,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max"),
        LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3))
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator