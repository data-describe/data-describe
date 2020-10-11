import hashlib
from functools import reduce
from typing import Optional, Union
import warnings

from data_describe.backends import _get_compute_backend
from data_describe.compat import _DATAFRAME_TYPE, _compat, _requires
from data_describe.config._config import get_option
from data_describe._widget import BaseWidget


_DEFAULT_SCORE_THRESHOLD = get_option("sensitive_data.score_threshold")
_SAMPLE_SIZE = get_option("sensitive_data.sample_size")


def sensitive_data(
    df,
    mode: str = "redact",
    detect_infotypes: bool = True,
    columns: Optional[list] = None,
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
    sample_size: int = _SAMPLE_SIZE,
    engine_backend=None,
    compute_backend: Optional[str] = None,
):
    """Identifies, redacts, and/or encrypts PII data.

    Note:
        `sensitive_data` uses Microsoft's Presidio in the backend. Presidio can be used
        to help identify sensitive data. However, because Presidio uses trained ML models,
        there is no guarantee that Presidio will find all sensitive information.

    Args:
        df (DataFrame): The dataframe
        mode (str): {'redact', 'encrypt'}
            redact: Redact the sensitive data
            encrypt: Anonymize the sensitive data
        detect_infotypes (bool): If True, identifies infotypes for each column
        columns ([str]): Defaults to None
        score_threshold (float): Minimum confidence value for detected entities to be returned. Default is 0.2.
        sample_size (int): Number of sampled rows used for identifying column infotypes. Default is 100.
        engine_backend: The backend analyzer engine. Default is presidio_analyzer.
        compute_backend (str): Select compute backend

    Raises:
        ValueError: Invalid input data type.
        TypeError: `columns` not a list of strings.

    Returns:
        SensitiveDataWidget
    """
    if not engine_backend:
        from data_describe.privacy.engine import engine  # noqa: lazy import

        engine_backend = engine

    if not isinstance(df, _DATAFRAME_TYPE):
        raise ValueError("Pandas data frame or modin data frame required")

    if _compat.check_install("modin.pandas"):
        if isinstance(df, _DATAFRAME_TYPE.modin):
            warnings.warn(
                "Sensitive data does not currently support Modin DataFrames. Converting to Pandas."
            )
            df = df._to_pandas()

    if columns:
        if not isinstance(columns, list):
            raise TypeError("cols must be type list")

    if mode not in ["encrypt", "redact", None]:
        raise ValueError("mode must be set to 'encrypt', 'redact', or None")

    sensitivewidget = _get_compute_backend(compute_backend, df).compute_sensitive_data(
        df=df,
        mode=mode,
        detect_infotypes=detect_infotypes,
        columns=columns,
        score_threshold=score_threshold,
        sample_size=sample_size,
        engine_backend=engine_backend,
    )

    sensitivewidget.columns = columns
    sensitivewidget.score_threshold = score_threshold
    sensitivewidget.sample_size = sample_size if detect_infotypes else None
    sensitivewidget.engine = engine_backend

    return sensitivewidget


class SensitiveDataWidget(BaseWidget):
    """Interface for collecting additional information about the sensitive data widget."""

    def __init__(
        self,
        engine=None,
        redact=None,
        encrypt=None,
        infotypes=None,
        sample_size=None,
        **kwargs,
    ):
        super(SensitiveDataWidget, self).__init__(**kwargs)
        self.engine = engine
        self.redact = redact
        self.encrypt = encrypt
        self.infotypes = infotypes
        self.sample_size = sample_size

    def __str__(self):
        return "data-describe Sensitive Data Widget"

    def show(self, **kwargs):
        """Show the transformed data or infotypes."""
        if isinstance(self.encrypt, _DATAFRAME_TYPE):
            viz_data = self.encrypt

        elif isinstance(self.redact, _DATAFRAME_TYPE):
            viz_data = self.redact

        elif self.infotypes:
            viz_data = self.infotypes

        return viz_data


@_requires("presidio_analyzer")
def compute_sensitive_data(
    df,
    mode: str = "redact",
    detect_infotypes: bool = True,
    columns: Optional[list] = None,
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
    sample_size: Union[int, float] = _SAMPLE_SIZE,
    engine_backend=None,
):
    """Identifies, redacts, and encrypts PII data.

    Note: sensitive_data uses Microsoft's Presidio in the backend. Presidio can be help identify sensitive data.
    However, because Presidio uses trained ML models, there is no guarantee that Presidio will find all sensitive information.

    Args:
        df (DataFrame): The dataframe
        mode (str): {'redact', 'encrypt'}
            redact: Redact the sensitive data
            encrypt: Anonymize the sensitive data
        detect_infotypes (bool): If True, identifies infotypes for each column
        columns ([str]): Defaults to None
        score_threshold (float): Minimum confidence value for detected entities to be returned. Default is 0.2.
        sample_size (int): Number of sampled rows used for identifying column infotypes. Default is 100.
        engine_backend: The backend analyzer engine. Default is presidio_analyzer.

    Raises:
        ValueError: `sample_size` greater than data size.

    Returns:
        SensitiveDataWidget
    """
    import data_describe.privacy.detection as ddsensitive  # noqa: Lazy load

    if columns:
        df = df[columns]

    sensitivewidget = ddsensitive.SensitiveDataWidget()

    if detect_infotypes:
        if sample_size > len(df):
            raise ValueError(f"sample_size must be less than {len(df)}")
        infotypes = identify_infotypes(df, engine_backend, sample_size)
        sensitivewidget.infotypes = infotypes

    if mode == "redact":
        df = df.applymap(lambda x: redact_info(str(x), engine_backend, score_threshold))
        sensitivewidget.redact = df

    elif mode == "encrypt":
        df = df.applymap(
            lambda x: encrypt_text(str(x), engine_backend, score_threshold)
        )
        sensitivewidget.encrypt = df

    return sensitivewidget


def identify_pii(text, engine_backend, score_threshold=_DEFAULT_SCORE_THRESHOLD):
    """Identifies infotypes contained in a string.

    Args:
        text (str): A string value
        engine_backend: The backend analyzer engine. Default is presidio_analyzer.
        score_threshold (float): Minimum confidence value for detected entities to be returned

    Returns:
        List of presidio_analyzer.recognizer_result.RecognizerResult
    """
    response = engine_backend.analyze(
        correlation_id=0,
        text=str(text).lower(),
        entities=[],
        language="en",
        all_fields=True,
        score_threshold=score_threshold,
    )
    return response


def create_mapping(text, response):
    """Identifies sensitive data and creates a mapping with the hashed data.

    Args:
        text (str): String value
        response: List of presidio_analyzer.recognizer_result.RecognizerResult

    Returns:
        word_mapping (dict): Mapping of the hashed data with the redacted string
        ref_text (str): String with hashed values
    """
    ref_text = text
    word_mapping = {}
    for r in response:
        hashed = hash_string(text[r.start : r.end])
        word_mapping[hashed] = str("<" + r.entity_type + ">")
        ref_text = ref_text.replace(text[r.start : r.end], hashed)
    return word_mapping, ref_text


def redact_info(text, engine_backend, score_threshold=_DEFAULT_SCORE_THRESHOLD):
    """Redact sensitive data with mapping between hashed values and infotype.

    Args:
        text (str): String value
        engine_backend: The backend analyzer engine. Default is presidio_analyzer.
        score_threshold (float): Minimum confidence value for detected entities to be returned

    Returns:
        String with redacted information
    """
    response = identify_pii(text, engine_backend, score_threshold)
    word_mapping, text = create_mapping(text, response)
    return reduce(lambda a, kv: a.replace(*kv), word_mapping.items(), text)


def identify_column_infotypes(
    data_series,
    engine_backend,
    sample_size: Union[int, float] = _SAMPLE_SIZE,
    score_threshold=_DEFAULT_SCORE_THRESHOLD,
):
    """Identifies the infotype of a pandas series object using a sample of rows.

    Args:
        data_series (Series): A Series
        engine_backend: The backend analyzer engine. Default is presidio_analyzer.
        sample_size (int): Number of rows to sample from
        score_threshold (float): Minimum confidence value for detected entities to be returned

    Returns:
        List of infotypes
    """
    if isinstance(sample_size, int):
        sampled_data = data_series.sample(n=sample_size, random_state=1)
    elif isinstance(sample_size, float):
        sampled_data = data_series.sample(frac=sample_size, random_state=1)
    results = list(
        sampled_data.map(
            lambda x: identify_pii(
                text=x, engine_backend=engine_backend, score_threshold=score_threshold
            )
        )
    )
    if results:
        return sorted(list(set([i.entity_type for obj in results for i in obj])))


def identify_infotypes(
    df,
    engine_backend,
    sample_size=_SAMPLE_SIZE,
    score_threshold=_DEFAULT_SCORE_THRESHOLD,
):
    """Identify infotypes for each column in the dataframe.

    Args:
        df (DataFrame): The dataframe
        engine_backend: The backend analyzer engine. Default is presidio_analyzer.
        sample_size (int): Number of rows to sample from
        score_threshold (float): Minimum confidence value for detected entities to be returned

    Returns:
        Dictionary with columns as keys and values as infotypes detected
    """
    return {
        col: identify_column_infotypes(
            df[col],
            engine_backend=engine_backend,
            sample_size=sample_size,
            score_threshold=score_threshold,
        )
        for col in df.columns
    }


def encrypt_text(text, engine_backend, score_threshold=_DEFAULT_SCORE_THRESHOLD):
    """Encrypt text using python's hash function.

    Args:
        text (str): A string value
        engine_backend: The backend analyzer engine. Default is presidio_analyzer.
        score_threshold (float): Minimum confidence value for detected entities to be returned

    Returns:
        Text with hashed sensitive data
    """
    response = identify_pii(text, engine_backend, score_threshold)
    return create_mapping(text, response)[1]


def hash_string(text):
    """Applies SHA256 text hashing.

    Args:
        text (str): The string value

    Returns:
        sha_signature: Hashed text
    """
    sha_signature = hashlib.sha256(text.encode()).hexdigest()
    return sha_signature
