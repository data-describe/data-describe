import hashlib
from functools import reduce
from typing import Optional, Any, Union

from presidio_analyzer import AnalyzerEngine

from data_describe.config._config import set_config, get_option


set_config(
    {
        "sensitive_data.score_threshold": 0.2,
        "sensitive_data.enable_trace_pii": True,
        "sensitive_data.sample_size": 100,
    }
)
_DEFAULT_SCORE_THRESHOLD = get_option("sensitive_data.score_threshold")
_ENABLE_TRACE_PII = get_option("sensitive_data.enable_trace_pii")
_SAMPLE_SIZE: int = get_option("sensitive_data.sample_size")


engine = AnalyzerEngine(
    default_score_threshold=_DEFAULT_SCORE_THRESHOLD, enable_trace_pii=_ENABLE_TRACE_PII
)


def compute_sensitive_data(
    df,
    redact: bool = True,
    encrypt: bool = False,
    detect_infotypes: bool = False,
    cols: Optional[list] = None,
    score_threshold: Optional[float] = None,
    sample_size: Optional[int] = None,
) -> Union[Any, dict]:
    """Identifies, redacts, and encrypts PII data
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
    score_threshold = score_threshold or _DEFAULT_SCORE_THRESHOLD
    sample_size = sample_size or _SAMPLE_SIZE
    if cols:
        df = df[cols]
    if redact:
        df = df.applymap(lambda x: redact_info(str(x), score_threshold))
    if detect_infotypes:
        if sample_size > len(df):
            raise ValueError(f"sample_size must be <= {len(df)}")
        df = identify_infotypes(df, sample_size)
    if encrypt:
        df = df.applymap(lambda x: encrypt_text(str(x), score_threshold))
    return df


def identify_pii(text, score_threshold=None):
    """Identifies infotypes contained in a string

    Args:
        text: A string value
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        List of presidio_analyzer.recognizer_result.RecognizerResult
        Results contain infotype, position, and confidence
    """
    score_threshold = score_threshold or _DEFAULT_SCORE_THRESHOLD
    response = engine.analyze(
        correlation_id=0,
        text=str(text).lower(),
        entities=[],
        language="en",
        all_fields=True,
        score_threshold=score_threshold,
    )
    return response


def create_mapping(text, response):
    """Identifies sensitive data and creates a mapping with the hashed data

    Args:
        text: String value
        response: List of presidio_analyzer.recognizer_result.RecognizerResult

    Returns:
        word_mapping: Mapping of the hashed data with the redacted string
        ref_text: String with hashed values
    """
    ref_text = text
    word_mapping = {}
    for r in response:
        hashed = hash_string(text[r.start : r.end])
        word_mapping[hashed] = str("<" + r.entity_type + ">")
        ref_text = ref_text.replace(text[r.start : r.end], hashed)
    return word_mapping, ref_text


def redact_info(text, score_threshold=None):
    """Redact sensitive data with mapping between hashed values and infotype

    Args:
        text: String value
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        String with redacted information
    """
    score_threshold = score_threshold or _DEFAULT_SCORE_THRESHOLD
    response = identify_pii(text, score_threshold)
    word_mapping, text = create_mapping(text, response)
    return reduce(lambda a, kv: a.replace(*kv), word_mapping.items(), text)


def identify_column_infotypes(data_series, sample_size=None, score_threshold=None):
    """Identifies the infotype of a pandas series object using a sample of rows

    Args:
        data_series: A pandas series
        sample_size: Number of rows to sample from
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        List of infotypes detecteds
    """
    score_threshold = score_threshold or _DEFAULT_SCORE_THRESHOLD
    sample_size = sample_size or _SAMPLE_SIZE

    sampled_data = data_series.sample(sample_size, random_state=1)
    results = sampled_data.map(
        lambda x: identify_pii(text=x, score_threshold=score_threshold)
    ).tolist()
    if results:
        return sorted(list(set([i.entity_type for obj in results for i in obj])))


def identify_infotypes(df, sample_size=None, score_threshold=None):
    """Identify infotypes for each column in the dataframe

    Args:
        df: The dataframe
        sample_size: Number of rows to sample from
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        Dictionary with columns as keys and values as infotypes detected
    """
    score_threshold = score_threshold or _DEFAULT_SCORE_THRESHOLD
    sample_size = sample_size or _SAMPLE_SIZE

    return {
        col: identify_column_infotypes(
            df[col], sample_size=sample_size, score_threshold=score_threshold
        )
        for col in df.columns
    }


def encrypt_text(text, score_threshold=None):
    """Encrypt text using python's hash function

    Args:
        text: A string value
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        Text with hashed sensitive data
    """
    score_threshold = score_threshold or _DEFAULT_SCORE_THRESHOLD

    response = identify_pii(text, score_threshold)
    return create_mapping(text, response)[1]


def hash_string(text):
    """Applies SHA256 text hashing

    Args:
        text: The string value

    Returns:
        sha_signature: Salted hash of the text
    """
    sha_signature = hashlib.sha256(text.encode()).hexdigest()
    return sha_signature
