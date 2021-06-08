import pdb

import fastestimator as fe
import numpy as np

x = np.random.rand(100, 10)
y = [0] * 10 + [1] * 90
location = [0] * 20 + [1] * 80

dataset = fe.dataset.NumpyDataset(data={"x": x, "y": y, "loc": location})

# stratified splitting
ds3 = dataset.split(0.1, stratify="loc")
print(ds3["loc"])
