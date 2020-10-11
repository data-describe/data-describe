import pytest

from data_describe.compat import _compat, _is_dataframe
from data_describe.privacy.detection import (
    sensitive_data,
    SensitiveDataWidget,
    presidio_engine,
)
from data_describe.privacy.detection import (
    identify_pii,
    redact_info,
    encrypt_text,
    hash_string,
)


def test_senstive_data_widget():
    sd = SensitiveDataWidget()
    assert hasattr(sd, "engine"), "Sensitive Data Widget missing engine"
    assert hasattr(sd, "redact"), "Sensitive Data Widget missing redact"
    assert hasattr(sd, "encrypt"), "Sensitive Data Widget missing encrypt"
    assert hasattr(sd, "infotypes"), "Sensitive Data Widget missing infotypes"
    assert hasattr(sd, "sample_size"), "Sensitive Data Widget missing sample size"
    assert hasattr(sd, "__repr__"), "Sensitive Data Widget missing __repr__ method"
    assert hasattr(
        sd, "_repr_html_"
    ), "Sensitive Data Widget missing _repr_html_ method"
    assert hasattr(sd, "show"), "Sensitive Data Widget missing show method"


def test_identify_pii():
    example_text = "This string contains a domain, gmail.com"
    response = identify_pii(example_text, presidio_engine())
    assert isinstance(response, list)
    assert isinstance(
        response[0], _compat["presidio_analyzer"].recognizer_result.RecognizerResult
    )
    assert len(response) == 1
    assert isinstance(response[0].entity_type, str)
    assert isinstance(response[0].start, int)
    assert isinstance(response[0].end, int)
    assert isinstance(response[0].score, float)
    assert response[0].entity_type == "DOMAIN_NAME"


def test_redact_info():
    example_text = "This string contains a domain gmail.com"
    result_text = redact_info(example_text, presidio_engine())
    assert isinstance(result_text, str)
    assert example_text != result_text
    assert result_text == "This string contains a domain <DOMAIN_NAME>"


def test_sensitive_data_cols(compute_backend_pii_df):
    sensitivewidget = sensitive_data(
        compute_backend_pii_df, mode="redact", columns=["name"], detect_infotypes=False
    )
    assert isinstance(sensitivewidget, SensitiveDataWidget)
    assert _is_dataframe(sensitivewidget.redact)
    assert sensitivewidget.redact.shape == (1, 1)
    assert isinstance(sensitivewidget.infotypes, type(None))
    assert isinstance(sensitivewidget.encrypt, type(None))


def test_only_redact_data(compute_backend_pii_df):
    sensitivewidget = sensitive_data(
        compute_backend_pii_df, mode="redact", detect_infotypes=False
    )
    assert isinstance(sensitivewidget, SensitiveDataWidget)
    assert sensitivewidget.redact.shape == (1, 2)
    assert _is_dataframe(sensitivewidget.redact)
    assert sensitivewidget.redact["name"][1] == "<PERSON>"
    assert sensitivewidget.redact["domain"][1] == "<DOMAIN_NAME>"
    assert isinstance(sensitivewidget.redact["name"][1], str)
    assert isinstance(sensitivewidget.redact["domain"][1], str)
    assert isinstance(sensitivewidget.encrypt, type(None))
    assert isinstance(sensitivewidget.infotypes, type(None))


def test_only_encrypt_data(compute_backend_pii_df):
    sensitivewidget = sensitive_data(
        compute_backend_pii_df, mode="encrypt", detect_infotypes=False
    )
    assert isinstance(sensitivewidget, SensitiveDataWidget)
    assert _is_dataframe(sensitivewidget.encrypt)
    assert isinstance(sensitivewidget.encrypt["name"][1], str)
    assert isinstance(sensitivewidget.encrypt["domain"][1], str)
    assert isinstance(sensitivewidget.redact, type(None))
    assert isinstance(sensitivewidget.infotypes, type(None))


def test_redact_data_and_infotypes(compute_backend_pii_df):
    sensitivewidget = sensitive_data(
        compute_backend_pii_df, mode="redact", detect_infotypes=True, sample_size=1
    )
    assert sensitivewidget.redact["name"][1] == "<PERSON>"
    assert sensitivewidget.redact["domain"][1] == "<DOMAIN_NAME>"
    assert isinstance(sensitivewidget.redact["name"][1], str)
    assert isinstance(sensitivewidget.redact["domain"][1], str)
    assert isinstance(sensitivewidget.infotypes, dict)
    assert sensitivewidget.infotypes["domain"][0] == "DOMAIN_NAME"
    assert sensitivewidget.infotypes["name"][0] == "PERSON"
    assert len(sensitivewidget.infotypes) == 2
    assert isinstance(sensitivewidget.encrypt, type(None))


def test_encrypt_data_and_infotypes(compute_backend_pii_df):
    sensitivewidget = sensitive_data(
        compute_backend_pii_df, mode="encrypt", detect_infotypes=True, sample_size=1
    )
    assert isinstance(sensitivewidget.encrypt["name"][1], str)
    assert isinstance(sensitivewidget.encrypt["domain"][1], str)
    assert isinstance(sensitivewidget.infotypes, dict)
    assert sensitivewidget.infotypes["domain"][0] == "DOMAIN_NAME"
    assert sensitivewidget.infotypes["name"][0] == "PERSON"
    assert len(sensitivewidget.infotypes) == 2
    assert isinstance(sensitivewidget.redact, type(None))
    assert _is_dataframe(sensitivewidget.show())


def test_encrypt_text():
    text = "gmail.com"
    encrypted = encrypt_text(text, presidio_engine())
    assert text != encrypted
    assert isinstance(encrypted, str)


def test_hash_string():
    hashed = hash_string("John Doe")
    assert isinstance(hashed, str)
    assert len(hashed) == 64


def test_type_df_type(compute_backend_pii_text):
    with pytest.raises(ValueError):
        sensitive_data(compute_backend_pii_text)


def test_column_type(compute_backend_pii_df):
    with pytest.raises(TypeError):
        sensitive_data(compute_backend_pii_df, columns="this is not a list")


def test_mode_value(compute_backend_pii_df):
    with pytest.raises(ValueError):
        sensitive_data(compute_backend_pii_df, mode="invalid mode")


def test_sample_size(compute_backend_pii_df):
    with pytest.raises(ValueError):
        sensitive_data(
            compute_backend_pii_df, mode=None, detect_infotypes=True, sample_size=9
        )
