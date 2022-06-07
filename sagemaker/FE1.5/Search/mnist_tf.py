import os

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.search import GridSearch
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver, RestoreWizard
from fastestimator.trace.metric import Accuracy


def get_estimator(init_lr, batch_size, save_dir, restore_dir):
    # step 1
    train_data, eval_data = mnist.load_data()
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
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
        LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=init_lr)),
        RestoreWizard(directory=restore_dir)
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=2, traces=traces)
    return estimator


def score_fn(search_idx, init_lr, batch_size, save_dir, restore_dir):
    est = get_estimator(init_lr=init_lr,
                        restore_dir=os.path.join(restore_dir, str(search_idx)),
                        save_dir=os.path.join(save_dir, str(search_idx)),
                        batch_size=batch_size)
    hist = est.fit(summary="exp")
    best_acc = float(max(list(hist.history["eval"]["max_accuracy"].values())))
    return best_acc


def fastestimator_run(restore_dir, save_dir):
    score_fn_in_use = lambda search_idx, init_lr, batch_size: score_fn(search_idx, init_lr, batch_size, save_dir=save_dir, restore_dir=restore_dir)
    mnist_grid_search = GridSearch(eval_fn=score_fn_in_use,
                                   params={
                                       "batch_size": [32, 64], "init_lr": [1e-2, 1e-3]
                                   },
                                   best_mode="max")
    mnist_grid_search.fit(save_dir=restore_dir)
