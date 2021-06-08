import fastestimator as fe
import numpy as np

x = np.random.rand(100, 10)
y = [0] * 10 + [1] * 90

ds = fe.dataset.NumpyDataset(data={"x": x, "y": y})

# 1 deterministic splitting
ds2 = ds.split(0.2, seed=0)
print(ds2["y"])
