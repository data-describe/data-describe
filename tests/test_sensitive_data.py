import pandas as pd
import pytest
import presidio_analyzer

from data_describe.sensitive_data.sensitive_data import sensitive_data
from data_describe.backends.compute._pandas.sensitive_data import (
    identify_pii,
    create_mapping,
    redact_info,
    identify_column_infotypes,
    identify_infotypes,
    encrypt_text,
    hash_string,
)


def test_identify_pii():
    example_text = "This string contains a domain, gmail.com"
    response = identify_pii(example_text)
    assert isinstance(response, list)
    assert isinstance(response[0], presidio_analyzer.recognizer_result.RecognizerResult)
    assert len(response) == 1
    assert isinstance(response[0].entity_type, str)
    assert isinstance(response[0].start, int)
    assert isinstance(response[0].end, int)
    assert isinstance(response[0].score, float)
    assert response[0].entity_type == "DOMAIN_NAME"


def test_identify_column_infotypes():
    test_series = pd.Series(["This string contains a domain, gmail.com"])
    results = identify_column_infotypes(test_series, sample_size=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], str)
    assert results[0] == "DOMAIN_NAME"


def test_identify_infotypes():
    df = pd.DataFrame({"domain": "gmail.com", "name": "John Doe"}, index=[1])
    results = identify_infotypes(df, sample_size=1)
    assert isinstance(results, dict)
    assert len(results) == 2
    assert isinstance(results["domain"], list)
    assert isinstance(results["name"], list)
    assert results["domain"][0] == "DOMAIN_NAME"
    assert results["name"][0] == "PERSON"


def test_create_mapping():
    example_text = "This string contains a domain gmail.com"
    response = identify_pii(example_text)
    word_mapping, text = create_mapping(example_text, response)
    assert isinstance(word_mapping, dict)
    assert isinstance(text, str)
    assert example_text != text


def test_redact_info():
    example_text = "This string contains a domain gmail.com"
    result_text = redact_info(example_text)
    assert isinstance(result_text, str)
    assert example_text != result_text
    assert result_text == "This string contains a domain <DOMAIN_NAME>"


def test_sensitive_data_cols(compute_backend):
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
        sensitive_data(
            pd.DataFrame(), redact=False, detect_infotypes=True, sample_size=1
        )
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


def test_encrypt_text():
    text = "gmail.com"
    encrypted = encrypt_text(text)
    assert text != encrypted
    assert isinstance(encrypted, str)


def test_encrypt_data():
    df = pd.DataFrame({"domain": "gmail.com", "name": "John Doe"}, index=[1])
    encrypted_df = sensitive_data(df, redact=False, encrypt=True)
    assert isinstance(encrypted_df, pd.core.frame.DataFrame)
    assert isinstance(encrypted_df.loc[1, "name"], str)
    assert isinstance(encrypted_df.loc[1, "domain"], str)


def test_hash_string():
    hashed = hash_string("John Doe")
    assert isinstance(hashed, str)
    assert len(hashed) == 64
