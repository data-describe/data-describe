import mwdata as mw
import pandas as pd
import numpy as np
import pytest
import matplotlib
from sklearn.preprocessing import LabelEncoder
matplotlib.use('Agg')


@pytest.fixture
def data():
    df = pd.read_csv('data/er_data.csv')
    df = df.sample(5, axis=1, random_state=3)
    return df


def test_importance(data):
    importance_vals = mw.importance(data, data.columns.values[0], return_values=True)
    assert len(importance_vals) == data.shape[1] - 1


def test_importance_num_only(data):
    df = pd.read_csv('data/er_data.csv')
    data = df.select_dtypes(['number'])
    assert isinstance(mw.importance(data, data.columns.values[0], return_values=True), np.ndarray)


def test_importance_cat_only(data):
    num_columns = data.select_dtypes(['number']).columns.values
    data = data[[c for c in data.columns if c not in num_columns]]
    assert len(mw.importance(data, data.columns.values[0], return_values=True)) == data.shape[1] - 1


def test_importance_preprocess(data):
    def pre(df, target):
        y = df[target]
        df = df.drop(target, axis=1)
        x_num = df.select_dtypes(['number'])
        x_num = x_num.fillna("-1")
        x_cat = df[[c for c in df.columns if not c in x_num.columns]].astype(str)
        x_cat = x_cat.fillna("")
        x_cat_encoded = x_cat.apply(LabelEncoder().fit_transform)
        X = pd.concat([x_num, x_cat_encoded], axis=1)
        return X, y

    fig = mw.importance(data, data.columns.values[0], preprocess_func=pre)
    assert isinstance(fig, matplotlib.artist.Artist)
