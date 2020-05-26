import sys
sys.path.append('../')
import notebooks

import glob
import importlib_resources
from pytest_notebook.nb_regression import NBRegressionFixture


fixture = NBRegressionFixture(exec_timeout=120, diff_color_words=True, diff_ignore=('/cells/*/outputs/*/data/text/plain',))


# def test_scatter_plots_notebook():
#     with importlib_resources.path(package=notebooks, resource='Scatter Plots.ipynb') as path:
#         result = fixture.check(str(path), raise_errors=True)


# def test_auto_data_type_notebook():
#     with importlib_resources.path(package=notebooks, resource='Auto Data Type.ipynb') as path:
#         result = fixture.check(str(path), raise_errors=True)


def test_cluster_analysis_notebook():
    with importlib_resources.path(package=notebooks, resource='Cluster Analysis.ipynb') as path:
        result = fixture.check(str(path), raise_errors=True)


# def test_correlation_matrix_notebook():
#     with importlib_resources.path(package=notebooks, resource='Correlation Matrix.ipynb') as path:
#         result = fixture.check(str(path), raise_errors=True)


# def test_data_heatmap_notebook():
#     with importlib_resources.path(package=notebooks, resource='Data Heatmap.ipynb') as path:
#         result = fixture.check(str(path), raise_errors=True)


# def test_data_loader_notebook():
#     with importlib_resources.path(package=notebooks, resource='Data Loader.ipynb') as path:
#         result = fixture.check(str(path), raise_errors=True)
