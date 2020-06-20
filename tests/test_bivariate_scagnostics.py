# import pandas as pd
# import data_describe as mw
# import matplotlib
# import pytest

# matplotlib.use("Agg")


# @pytest.fixture
# def data():
#     file_path = "data/er_data.csv"
#     df = pd.read_csv(file_path)
#     num_df = df.select_dtypes(["number"]).iloc[:, [0, 1, 2, 3]]
#     return num_df.sample(250, random_state=1)


# @pytest.fixture
# def np_data():
#     file_path = "data/er_data.csv"
#     df = pd.read_csv(file_path)
#     num_df = df.select_dtypes(["number"]).iloc[:, [0, 1, 2, 3]]
#     return num_df.sample(250, random_state=1).values


# @pytest.fixture
# def scagnostic_class(data):
#     return mw.metrics.bivariate.Scagnostics(data)


# @pytest.fixture
# def scagnostics_default_results(scagnostic_class):
#     return scagnostic_class.calculate()


# @pytest.fixture
# def scagnostics_graph_results(scagnostic_class):
#     return scagnostic_class.calculate(graphs=True)


# def test_scagnostic_class_np(np_data):
#     assert isinstance(
#         mw.metrics.bivariate.Scagnostics(np_data), mw.metrics.bivariate.Scagnostics
#     )


# def test_str_data_type():
#     assert isinstance(
#         mw.metrics.bivariate.Scagnostics("data/er_data.csv"),
#         mw.metrics.bivariate.Scagnostics,
#     )


# def test_calculate_metrics_no_data(scagnostic_class):
#     with pytest.raises(ValueError):
#         scagnostic_class.calculate_metrics((0, 1))


# def test_scagnostics_default(scagnostics_default_results, data):
#     c = len(data.select_dtypes(["number"]).columns.values)
#     assert len(scagnostics_default_results) == c * (c - 1) / 2


# def test_scagnostics_serial(scagnostic_class, scagnostics_default_results):
#     result = scagnostic_class.calculate(parallel=False)
#     assert result == scagnostics_default_results


# def test_names(scagnostic_class):
#     assert set(scagnostic_class.names) == {
#         "time_in_hospital",
#         "num_lab_procedures",
#         "num_procedures",
#         "num_medications",
#     }
