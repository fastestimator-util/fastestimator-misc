import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.gradient import GradientOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp


def get_estimator():
    # step 1
    train_data, eval_data = mnist.load_data()
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           batch_size=32,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])
    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce", merge_grad=4)
    ])
    # step 3
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=2)
    return estimator
