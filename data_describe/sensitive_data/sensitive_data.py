from data_describe.backends import _get_compute_backend
from data_describe.compat import _DATAFRAME_TYPE


def sensitive_data(
    df,
    redact=True,
    encrypt=False,
    detect_infotypes=False,
    cols=None,
    compute_backend=None,
    **kwargs
):
    """Identifies, redacts, and encrypts PII data.

    Note: sensitive_data uses Microsoft's Presidio in the backend. Presidio can be help identify sensitive data.
    However, because Presidio uses trained ML models, there is no guarantee that Presidio will find all sensitive information.

    Args:
        df: The dataframe
        redact: If True, redact sensitive data
        encrypt: If True, anonymize data. Redact must be set to False
        detect_infotypes: If True, identifies infotypes for each column. Redact must be set to False
        score_threshold: Minimum confidence value for detected entities to be returned
        sample_size: Number of sampled rows used for identifying column infotypes
        cols: List of columns. Defaults to None

    Returns:
        A dataframe if redact or anonymize is True.
        Dictionary of column infotypes if detect_infotypes is True
    """
    if not isinstance(df, _DATAFRAME_TYPE):
        raise TypeError("Pandas data frame or modin data frame required")

    x = _get_compute_backend(compute_backend, df)

    print("########################")
    print(x.b)
    df = _get_compute_backend(compute_backend, df).process_sensitive_data(
        df=df,
        redact=redact,
        encrypt=encrypt,
        detect_infotypes=detect_infotypes,
        cols=cols,
        **kwargs
    )

    return df
