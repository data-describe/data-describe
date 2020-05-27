from pytest_notebook.nb_regression import NBRegressionFixture
import importlib_resources
import notebooks


fixture = NBRegressionFixture(exec_timeout=120, diff_color_words=True, diff_ignore=('/cells/*/outputs/*/data/text/plain',
                                                                                    '/cells/*/outputs/*/data/image/png',
                                                                                    '/cells/*/outputs/*/data/text/html',
                                                                                    '/cells/*/outputs/*/data/application/vnd.plotly.v1+json'))


def test_geospatial_analysis_notebook():
    with importlib_resources.path(package=notebooks, resource='Geospatial Analysis.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


def test_model_evaluation_notebook():
    with importlib_resources.path(package=notebooks, resource='Model Evaluation.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


def test_data_loader_notebook():
    with importlib_resources.path(package=notebooks, resource='Data Loader.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


def test_text_preprocessing_notebook():
    with importlib_resources.path(package=notebooks, resource='Text Preprocessing.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


def test_cluster_analysis_notebook():
    with importlib_resources.path(package=notebooks, resource='Cluster Analysis.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


def test_correlation_matrix_notebook():
    with importlib_resources.path(package=notebooks, resource='Correlation Matrix.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


def test_data_heatmap_notebook():
    with importlib_resources.path(package=notebooks, resource='Data Heatmap.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


def test_feature_importance_notebook():
    with importlib_resources.path(package=notebooks, resource='Feature Importance.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


def test_distributions_notebook():
    with importlib_resources.path(package=notebooks, resource='Distributions.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


def test_topic_model_notebook():
    with importlib_resources.path(package=notebooks, resource='Topic Modeling.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


def test_scatter_plots_notebook():
    with importlib_resources.path(package=notebooks, resource='Scatter Plots.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


def test_data_summary_notebook():
    fixture = NBRegressionFixture(exec_timeout=120, diff_color_words=True, diff_ignore=('/cells/4/outputs/0/text',))
    with importlib_resources.path(package=notebooks, resource='Data Summary.ipynb') as path:
        fixture.check(str(path), raise_errors=True)


# def test_tutorial_notebook():
#     with importlib_resources.path(package=notebooks, resource='Tutorial.ipynb') as path:
#         fixture.check(str(path), raise_errors=True)


# def test_auto_data_type_notebook():
#     with importlib_resources.path(package=notebooks, resource='Auto Data Type.ipynb') as path:
#         fixture.check(str(path), raise_errors=True)
