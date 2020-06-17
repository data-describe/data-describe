from functools import reduce
import hashlib

import pandas as pd
import spacy
from presidio_analyzer import AnalyzerEngine


if not spacy.util.is_package("en_core_web_lg"):
    spacy.cli.download("en_core_web_lg")

engine = AnalyzerEngine(default_score_threshold=0.5, enable_trace_pii=True)


def sensitive_data(
    df,
    redact=True,
    encrypt=False,
    detect_infotypes=False,
    score_threshold=0.2,
    sample_size=100,
    cols=[],
):
    """Identifies, redacts, and encrypts PII data

    Args:
        df: The dataframe
        redact: If True, redact sensitive data
        encrypt: If True, anonymize data. Redact must be set to False
        detect_infotypes: If True, identifies infotypes for each column. Redact must be set to False
        score_threshold: Minimum confidence value for detected entities to be returned
        sample_size: Number of sampled rows used for identifying column infotypes
        cols: List of columns

    Returns:
        A dataframe if redact or anonymize is True.
        Dictionary of column infotypes if detect_infotypes is True
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Pandas data frame required")
    if not isinstance(cols, list):
        raise TypeError("cols must be type list")
    if cols:
        df = df[cols]
    if (encrypt or detect_infotypes) and redact:
        raise ValueError("Set redact=False to encrypt or detect_infotypes")
    if redact:
        return redact_df(df, score_threshold)
    if detect_infotypes:
        if sample_size > len(df):
            raise ValueError(f"sample_size must be <= {len(df)}")
        return identify_infotypes(df, sample_size)
    if encrypt:
        return encrypt_data(df)


def identify_pii(text, score_threshold=0.2):
    """Identifies infotypes contained in a string

    Args:
        text: A string value
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        List of presidio_analyzer.recognizer_result.RecognizerResult
        Results contain infotype, position, and confidence
    """
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


def redact_info(text, score_threshold=0.2):
    """Redact sensitive data using mapping between hashed values and infotype

    Args:
        text: A string
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        String with redacted information
    """
    response = identify_pii(text, score_threshold)
    word_mapping, text = create_mapping(text, response)
    return reduce(lambda a, kv: a.replace(*kv), word_mapping.items(), text)


def redact_df(df, score_threshold=0.2):
    """Redact sensitive data in a dataframe

    Args:
        df: The dataframe
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        Dataframe with redacted information
    """
    return df.applymap(lambda x: redact_info(str(x), score_threshold))


def identify_column_infotypes(data_series, sample_size=100, score_threshold=0.2):
    """Identifies the infotype of a pandas series object using a sample of rows

    Args:
        data_series: A pandas series
        sample_size: Number of rows to sample from
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        List of infotypes detecteds
    """
    sampled_data = data_series.sample(sample_size, random_state=1)
    results = sampled_data.map(
        lambda x: identify_pii(text=x, score_threshold=score_threshold)
    ).tolist()
    if results:
        return sorted(list(set([i.entity_type for obj in results for i in obj])))


def identify_infotypes(df, sample_size=100, score_threshold=0.2):
    """Identify infotypes for each column in the dataframe

    Args:
        df: The dataframe
        sample_size: Number of rows to sample from
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        Dictionary with columns as keys and values as infotypes detected
    """
    return {
        col: identify_column_infotypes(
            df[col], sample_size=sample_size, score_threshold=score_threshold
        )
        for col in df.columns
    }


def encrypt_text(text, score_threshold=0.2):
    """Encrypt text using python's hash function

    Args:
        text: A string value
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        Text with hashed sensitive data
    """
    response = identify_pii(text, score_threshold)
    return create_mapping(text, response)[1]


def encrypt_data(df, score_threshold=0.2):
    """Encrypt sensitive data in a dataframe

    Args:
        df: The dataframe
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        Dataframe with encrypted data
    """
    return df.applymap(lambda x: encrypt_text(str(x), score_threshold))


def hash_string(text):
    """Applies SHA256 text hashing

    Args:
        text: The string value

    Returns:
        sha_signature: Salted hash of the text
    """
    sha_signature = hashlib.sha256(text.encode()).hexdigest()
    return sha_signature
