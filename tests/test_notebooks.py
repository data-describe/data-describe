import os

from pytest_notebook.nb_regression import NBRegressionFixture


EXEC_CWD = os.path.abspath(".")

fixture = NBRegressionFixture(
    exec_timeout=120,
    exec_cwd=EXEC_CWD,
    diff_color_words=True,
    diff_ignore=(
        "/cells/*/outputs/*/data/text/plain",
        "/cells/*/metadata/outputs/text",
        "/metadata/*",
        "/cells/1/outputs",
        "/cells/*/outputs/*/text",
        "/cells/*/outputs/*/data/image/png",
        "/cells/*/outputs/*/data/text/html",
        "/cells/*/outputs/*/data/application/vnd.plotly.v1+json",
    ),
)


def test_cluster_analysis_notebook():
    fixture.check(
        os.path.join(EXEC_CWD, "notebooks", "Cluster_Analysis.ipynb"), raise_errors=True
    )


def test_text_preprocessing_notebook():
    fixture.check(
        os.path.join(EXEC_CWD, "notebooks", "Text_Preprocessing.ipynb"),
        raise_errors=True,
    )


def test_data_heatmap_notebook():
    fixture.check(
        os.path.join(EXEC_CWD, "notebooks", "Data_Heatmap.ipynb"), raise_errors=True
    )


def test_feature_importance_notebook():
    fixture.check(
        os.path.join(EXEC_CWD, "notebooks", "Feature_Importance.ipynb"),
        raise_errors=True,
    )


def test_correlation_matrix_notebook():
    fixture.check(
        os.path.join(EXEC_CWD, "notebooks", "Correlation_Matrix.ipynb"),
        raise_errors=True,
    )


def test_sensitive_data_notebook():
    fixture.check(
        os.path.join(EXEC_CWD, "notebooks", "Sensitive_Data.ipynb"), raise_errors=True
    )


def test_data_summary_notebook():
    fixture = NBRegressionFixture(
        exec_timeout=120,
        exec_cwd=EXEC_CWD,
        diff_color_words=True,
        diff_ignore=(
            "/cells/*/outputs/*/data/text/html",
            "/metadata/language_info/version",
            "/cells/4/outputs/0/text",
            "/cells/1/outputs/",
            "/cells/4/outputs/1/data/text/plain",
        ),
    )
    fixture.check(
        os.path.join(EXEC_CWD, "notebooks", "Data_Summary.ipynb"), raise_errors=True
    )


# LONG RUN TIME
# def test_scatter_plots_notebook():
#     fixture.check(
#         os.path.join(EXEC_CWD, "notebooks", "Scatter Plots.ipynb"),
#         raise_errors=True,
#     )


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
#         os.path.join(EXEC_CWD, "notebooks", "Topic Modeling.ipynb"),
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
#         os.path.join(EXEC_CWD, "notebooks", "Geospatial_Analysis.ipynb"),
#         raise_errors=True,
#     )
