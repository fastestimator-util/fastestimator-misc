import fastestimator as fe
from fastestimator.dataset.data.cifair10 import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize, Onehot

train_data, test_data = load_data()
pipeline = fe.Pipeline(
    train_data=train_data,
    test_data=test_data,
    batch_size=32,
    ops=[
        Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
        RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
        Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
        CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
        Onehot(inputs="y", outputs="y", mode="train", num_classes=10, label_smoothing=0.2)
    ])
pipeline.benchmark()
