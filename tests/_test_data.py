import pandas as pd
import numpy as np

np.random.seed(22)
DATA = pd.DataFrame(
    {
        "a": np.random.normal(2, 1.2, size=250),
        "b": np.random.normal(3, 1.5, size=250),
        "c": np.random.normal(9, 0.2, size=250),
        "d": np.random.choice(["x", "y"], size=250),
        "e": np.random.choice(["v", "w"], p=[0.01, 0.99], size=250),
    }
)
