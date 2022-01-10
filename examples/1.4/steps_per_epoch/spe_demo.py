import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import cifair10
from fastestimator.op.numpyop.univariate import Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy


def get_estimator(epochs=10, batch_size=32, train_steps_per_epoch=None):
    train_data, eval_data = cifair10.load_data()

    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           batch_size=batch_size,
                           ops=[Normalize(inputs="x", outputs="x")])

    # step 2
    model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    print("10 regular epochs")
    est = get_estimator(epochs=10, train_steps_per_epoch=1563)
    est.fit()
    print("\n---\n5 double epochs\n---\n")
    est = get_estimator(epochs=5, train_steps_per_epoch=3126)
    est.fit()
    # Time on DGX: 118 sec vs 79 sec

