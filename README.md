# data⎰describe
Expediting data discovery and understanding

## Install

With Pip:

```
pip install data-describe
```

With Conda:

```
conda install data-describe
```

## Usage

import

```python

import data_describe as dd

```

## Supported Backends
### Data / Computation
- Pandas
- Modin

### Visualization
- Seaborn (Matplotlib)
- Plotly

test with some data

```python

from sklearn.datasets import load_wine
data = load_wine()
df = pd.DataFrame(data.data, columns=list(data.feature_names))
df['target'] = data.target

dd.data_heatmap(df)

```

![heatmap](/docs/imgs/heatmap.png)





