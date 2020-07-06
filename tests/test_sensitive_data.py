import pandas as pd
import pytest

from data_describe.sensitive_data.sensitive_data import sensitive_data


def test_sensitive_data_cols():
    df = pd.DataFrame({"domain": "gmail.com", "name": "John Doe"}, index=[1])
    redacted_df = sensitive_data(df, redact=True, cols=["name"])
    assert redacted_df.shape == (1, 1)
    assert redacted_df.loc[1, "name"] == "<PERSON>"


def test_type():
    with pytest.raises(TypeError):
        sensitive_data("this is not a dataframe")
    with pytest.raises(TypeError):
        sensitive_data(pd.DataFrame(), cols="this is not a list")


def test_sample_size():
    with pytest.raises(ValueError):
        sensitive_data(pd.DataFrame(), redact=False, detect_infotypes=True)
    with pytest.raises(ValueError):
        sensitive_data(pd.DataFrame(), redact=True, encrypt=True)


def test_sensitive_data_redact():
    df = pd.DataFrame({"domain": "gmail.com", "name": "John Doe"}, index=[1])
    redacted_df = sensitive_data(df, redact=True, sample_size=1)
    assert redacted_df.shape == (1, 2)
    assert redacted_df.loc[1, "domain"] == "<DOMAIN_NAME>"
    assert redacted_df.loc[1, "name"] == "<PERSON>"
    assert isinstance(redacted_df, pd.core.frame.DataFrame)


def test_sensitive_data_detect_infotypes():
    df = pd.DataFrame({"domain": "gmail.com", "name": "John Doe"}, index=[1])
    results = sensitive_data(df, redact=False, detect_infotypes=True, sample_size=1)
    assert isinstance(results, dict)
    assert len(results) == 2
    assert isinstance(results["domain"], list)
    assert isinstance(results["name"], list)
    assert results["domain"][0] == "DOMAIN_NAME"
    assert results["name"][0] == "PERSON"
