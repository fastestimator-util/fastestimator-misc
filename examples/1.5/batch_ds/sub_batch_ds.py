import math
import pdb

import tensorflow as tf
from torch.utils.data import Dataset

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset import BatchDataset
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop import Batch
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop import LambdaOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp


class MyBatchDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, idx):
        image = self.ds[idx]["x"]
        label = self.ds[idx]["y"]
        return [{"x": image, "y": label} for _ in range(32)]

    def __len__(self):
        return len(self.ds)


class SubBatchDataset(Dataset):
    def __init__(self, batched_ds, batch_size=4):
        self.batched_ds = batched_ds
        self.batch_size = batch_size
        self.original_batch_size = self._get_old_batch_size()

    def _get_old_batch_size(self):
        return len(self.batched_ds[0])

    def _get_old_idx(self, idx):
        old_batch_idx = idx * self.batch_size // self.original_batch_size
        old_element_idx = (idx * self.batch_size) % self.original_batch_size
        return old_batch_idx, old_element_idx

    def __len__(self):
        return math.ceil(self.original_batch_size * len(self.batched_ds) / self.batch_size)

    def __getitem__(self, idx):
        old_batch_idx, old_element_idx = self._get_old_idx(idx)
        old_items = self.batched_ds[old_batch_idx]
        if old_element_idx + self.batch_size <= len(old_items):
            # if there is enough samples in the old batch
            new_batch = old_items[old_element_idx:old_element_idx + self.batch_size]
        else:
            if old_batch_idx + 1 < len(self.batched_ds):
                # if next batch still exists in old batch
                partial_batch = old_items[old_element_idx:]
                new_batch = partial_batch + self.batched_ds[old_batch_idx + 1][0:self.batch_size - len(partial_batch)]
            else:
                # when it is the last batch
                new_batch = old_items[old_element_idx:]
        return new_batch


def get_estimator(epochs=2):
    mnist_ds, _ = mnist.load_data()
    my_batch_ds = MyBatchDataset(mnist_ds)
    merged_batch_ds = BatchDataset(datasets=[mnist_ds, my_batch_ds], num_samples=[32, 1])
    final_ds = SubBatchDataset(merged_batch_ds, batch_size=13)
    pipeline = fe.Pipeline(
        train_data=final_ds,
        ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x"), Batch(batch_size=4)])
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs)
    return estimator
