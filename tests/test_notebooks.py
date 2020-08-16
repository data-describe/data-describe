import io
import os

from pytest_notebook.nb_regression import NBRegressionFixture
import papermill as pm
import nbformat

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


def check_notebook_execution(notebook):
    """Update notebook if cell execution counts are not ordered.

    Args:
        notebook: Path to notebook
    """
    with io.open(notebook) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)
    for idx, cell in enumerate(nb["cells"], 1):
        if "execution_count" in cell:
            if cell["execution_count"] != idx:
                pm.execute_notebook(notebook, notebook)
                break


def test_cluster_analysis_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Cluster_Analysis.ipynb")
    check_notebook_execution(notebook)
    fixture.check(notebook, raise_errors=True)


def test_text_preprocessing_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Text_Preprocessing.ipynb")
    check_notebook_execution(notebook)
    fixture.check(notebook, raise_errors=True)


def test_data_heatmap_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Data_Heatmap.ipynb")
    check_notebook_execution(notebook)
    fixture.check(notebook, raise_errors=True)


def test_feature_importance_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Feature_Importance.ipynb")
    check_notebook_execution(notebook)
    fixture.check(notebook, raise_errors=True)


def test_correlation_matrix_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Correlation_Matrix.ipynb")
    check_notebook_execution(notebook)
    fixture.check(notebook, raise_errors=True)


def test_data_summary_notebook():
    notebook = os.path.join(EXEC_CWD, "examples", "Data_Summary.ipynb")
    check_notebook_execution(notebook)
    # fixture = NBRegressionFixture(
    #     exec_timeout=120,
    #     exec_cwd=EXEC_CWD,
    #     diff_color_words=True,
    #     diff_ignore=(
    #         "/cells/*/outputs/*/data/text/html",
    #         "/cells/*/outputs/*/output/data/",
    #         "/cells/4/outputs/1/data/text/plain",
    #         "/cells/1/outputs/",
    #         "/cells/*/metadata/",
    #         "/cells/*/outputs/metadata/",
    #         "/metadata/",
    #         "/metadata/kernelspec/",
    #     ),
    # )
    fixture.check(notebook, raise_errors=True)


# LONG RUN TIME
# def test_scatter_plots_notebook():
#     notebook = os.path.join(EXEC_CWD, "examples", "Scatter_Plots.ipynb")
#     check_notebook_execution(notebook)
#     fixture.check(notebook, raise_errors=True)


# def test_topic_modeling():
#     fixture = NBRegressionFixture(
#         exec_timeout=120,
#         exec_cwd=EXEC_CWD,
#         diff_color_words=True,
#         diff_ignore=(
#             "/cells/*/outputs/*/data/text/html",
#             "/metadata/language_info/version",
#             "/cells/4/outputs/0/text",
#             "/cells/1/outputs/",
#             "/cells/7/outputs/0",
#             "/cells/4/outputs/1/data/text/plain",
#         ),
#     )
#     fixture.check(
#         os.path.join(EXEC_CWD, "examples", "Topic Modeling.ipynb"),
#         raise_errors=True,
#     )


# def test_scatter_plots_notebook():
#     fixture.check(os.path.join(EXEC_CWD, 'notebooks', 'Scatter Plots.ipynb'), raise_errors=True)


# def test_tutorial_notebook():
#     fixture.check(os.path.join(EXEC_CWD, 'notebooks', 'Tutorial.ipynb'), raise_errors=True)


# REQUIRES EXTERNAL DATA
# def test_data_loader_notebook():
#     fixture.check(os.path.join(EXEC_CWD, 'notebooks', 'Data Loader.ipynb'), raise_errors=True)


# def test_distributions_notebook():
#     fixture.check(os.path.join(EXEC_CWD, 'notebooks', 'Distributions.ipynb'), raise_errors=True)


# def test_auto_data_type_notebook():
#     fixture.check(os.path.join(EXEC_CWD, 'notebooks', 'Auto Data Type.ipynb'), raise_errors=True)


# def test_geospatial_analysis_notebook():
#     fixture.check(
#         os.path.join(EXEC_CWD, "examples", "Geospatial_Analysis.ipynb"),
#         raise_errors=True,
#     )
