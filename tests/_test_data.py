import pandas as pd
import numpy as np


DATA = pd.DataFrame(
    {
        "a": np.random.normal(2, 1.2, size=250),
        "b": np.random.normal(3, 1.5, size=250),
        "c": np.random.normal(9, 0.2, size=250),
        "d": np.random.choice(["x", "y"], size=250),
    }
)
