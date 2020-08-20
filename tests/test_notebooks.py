import os

from pytest_notebook.nb_regression import NBRegressionFixture

EXEC_CWD = os.path.abspath(".")

fixture = NBRegressionFixture(
    exec_timeout=120,
    exec_cwd=EXEC_CWD,
    diff_color_words=True,
    diff_ignore=(
        "/cells/*/outputs/*/data/text/plain",
        "/cells/*/outputs/*/data/image/svg+xml",
        "/cells/*/outputs/*/text",
        "/cells/*/outputs/*/data/image/png",
        "/cells/*/outputs/*/data/text/html",
        "/cells/*/outputs/*/output/data/"
        "/cells/*/outputs/*/data/application/vnd.plotly.v1+json",
        "/cells/*/outputs/*/execution_count",
        "/cells/*/execution_count",
        "/cells/1/outputs/",
        "/cells/*/metadata/",
        "/cells/*/outputs/metadata/",
        "/metadata/",
    ),
)


def test_cluster_analysis_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Cluster_Analysis.ipynb")
    fixture.check(notebook, raise_errors=True)


def test_data_summary_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Data_Summary.ipynb")
    fixture.check(notebook, raise_errors=True)


def test_text_preprocessing_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Text_Preprocessing.ipynb")
    fixture.check(notebook, raise_errors=True)


def test_topic_modeling_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Topic_Modeling.ipynb")
    fixture.check(notebook, raise_errors=True)


def test_data_heatmap_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Data_Heatmap.ipynb")
    fixture.check(notebook, raise_errors=True)


def test_feature_importance_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Feature_Importance.ipynb")
    fixture.check(notebook, raise_errors=True)


def test_correlation_matrix_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Correlation_Matrix.ipynb")
    fixture.check(notebook, raise_errors=True)


def test_scatter_plots_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Scatter_Plots.ipynb")
    fixture.check(notebook, raise_errors=True)


def test_tutorial_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Tutorial.ipynb")
    fixture.check(notebook, raise_errors=True)


# REQUIRES EXTERNAL DATA
# Data_Loader.ipynb
# Distributions.ipynb
