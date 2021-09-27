import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as fn
from sklearn.preprocessing import StandardScaler

import fastestimator as fe
from fastestimator.dataset.data import breast_cancer
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


class DNN(torch.nn.Module):
    def __init__(self, num_inputs=30, n_outputs=1):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        self.dp1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 16)
        self.dp2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(16, 8)
        self.dp3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(8, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = fn.relu(x)
        x = self.dp2(x)
        x = self.fc3(x)
        x = fn.relu(x)
        x = self.dp3(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x


def get_estimator(epochs=2,
                  batch_size=32,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp()):
    # step 1. prepare data
    train_data, eval_data = breast_cancer.load_data()

    # Apply some global pre-processing to the data
    scaler = StandardScaler()
    train_data["x"] = scaler.fit_transform(train_data["x"])
    eval_data["x"] = scaler.transform(eval_data["x"])

    pipeline = fe.Pipeline(train_data=train_data, eval_data=eval_data, batch_size=batch_size)

    # step 2. prepare model
    model = fe.build(model_fn=DNN, optimizer_fn="adam")

    class_weights = {0: 2.0, 1: 2.0}

    network = fe.Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce_weighted", class_weights=class_weights),
        UpdateOp(model=model, loss_name="ce")
    ])

    # step 3.prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             log_steps=10,
                             traces=traces,
                             monitor_names="ce_weighted",
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
