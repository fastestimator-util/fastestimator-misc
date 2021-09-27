import pdb

from torch.utils.data import Dataset

import fastestimator as fe
from fastestimator.dataset import BatchDataset
from fastestimator.dataset.data import mnist


class NegativeImageSimulatedTube(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, idx):
        # create your 5 simulated image here, for simplicity, I will just copy the same image 5 times
        image = self.ds[idx]["x"]
        label = self.ds[idx]["y"]
        return [{"x": image, "y": label} for _ in range(5)]

    def __len__(self):
        return len(self.ds)


def fastestimator_run():
    pos_real, _ = mnist.load_data(image_key="x1", label_key="y1")
    neg_real, _ = mnist.load_data(image_key="x2", label_key="y2")
    neg_sim, _ = mnist.load_data()
    neg_sim = NegativeImageSimulatedTube(neg_sim)
    batch_ds = BatchDataset(datasets=(pos_real, neg_real, neg_sim), num_samples=(10, 10, 2))
    pipeline = fe.Pipeline(train_data=batch_ds)
    data = pipeline.get_results()
    for key, value in data.items():
        print(key)
        print(value.shape)
