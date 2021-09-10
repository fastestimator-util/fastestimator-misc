"""
In this pretrain-finetune example, we do the following:

epoch 1 - 10: pretrain a ResNet9 model on Cifar100
epoch 11-12: freeze the backbone, use a new classification head and finetune on cifar10 data
"""
import tempfile

import fastestimator as fe
import torch
import torch.nn as nn
import torch.nn.functional as fn
from fastestimator.dataset.data import cifair10, cifair100
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule.schedule import EpochScheduler
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


class Backbone(nn.Module):
    def __init__(self, input_size=(3, 32, 32)):
        super().__init__()
        self.conv0 = nn.Conv2d(input_size[0], 64, 3, padding=(1, 1))
        self.conv0_bn = nn.BatchNorm2d(64, momentum=0.8)
        self.conv1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(128, momentum=0.8)
        self.residual1 = Residual(128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(256, momentum=0.8)
        self.residual2 = Residual(256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(512, momentum=0.8)
        self.residual3 = Residual(512)

    def forward(self, x):
        # prep layer
        x = self.conv0(x)
        x = self.conv0_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        # layer 1
        x = self.conv1(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual1(x)
        # layer 2
        x = self.conv2(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual2(x)
        # layer 3
        x = self.conv3(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv3_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual3(x)
        # layer 4
        x = nn.AdaptiveMaxPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        return x


class Residual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        return x


class Classifier(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        return self.fc(x)


class MyModel(nn.Module):
    def __init__(self, backbone, cls_head):
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head

    def forward(self, x):
        x = self.backbone(x)
        x = self.cls_head(x)
        return x


def get_estimator(epochs=12, batch_size=512, save_dir=tempfile.mkdtemp()):
    # epoch 1-10, train on cifair100,  epoch 11-end: train on cifar10
    cifair10_train, cifari10_test = cifair10.load_data()
    cifair100_train, _ = cifair100.load_data()
    train_ds = EpochScheduler({1: cifair100_train, 11: cifair10_train})
    pipeline = fe.Pipeline(
        train_data=train_ds,
        test_data=cifari10_test,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x")
        ])

    # step 2: prepare network
    backbone = fe.build(model_fn=lambda: Backbone(input_size=(3, 32, 32)), optimizer_fn="adam")
    cls_head_cifar100 = fe.build(model_fn=lambda: Classifier(classes=100), optimizer_fn="adam")
    cls_head_cifar10 = fe.build(model_fn=lambda: Classifier(classes=10), optimizer_fn="adam")
    # if you want to save the final cifar10 model, you can build a model then provide it to ModelSaver
    # final_model_cifar10 = fe.build(model_fn=lambda: MyModel(backbone, cls_head_cifar10), optimizer_fn=None)

    # epoch 1-10: train backbone and cls_head_cifar100, epoch 11-end: train cls_head_cifar10 only
    ModelOp_cls_head = EpochScheduler({
        1: ModelOp(model=cls_head_cifar100, inputs="feature", outputs="y_pred"),
        11: ModelOp(model=cls_head_cifar10, inputs="feature", outputs="y_pred"),
    })
    UpdateOp_backbone = EpochScheduler({1: UpdateOp(model=backbone, loss_name="ce"), 11: None})
    UpdateOp_cls_head = EpochScheduler({
        1: UpdateOp(model=cls_head_cifar100, loss_name="ce"), 11: UpdateOp(model=cls_head_cifar10, loss_name="ce")
    })
    network = fe.Network(ops=[
        ModelOp(model=backbone, inputs="x", outputs="feature"),
        ModelOp_cls_head,
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp_backbone,
        UpdateOp_cls_head
    ])
    traces = [Accuracy(true_key="y", pred_key="y_pred")]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
