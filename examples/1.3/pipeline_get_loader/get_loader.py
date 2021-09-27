import fastestimator as fe
from fastestimator.dataset.data.cifair10 import load_data

train_data, test_data = load_data()
pipeline = fe.Pipeline(train_data=train_data, test_data=test_data, batch_size=32)
"""
in FE 1.1, 1.2. To get the loader, you will do:
loader = pipeline.get_loader(mode=train)
"""

# Starting 1.3, you will do:
with pipeline(mode="train") as loader:
    for batch in loader:
        print(batch)
        break
