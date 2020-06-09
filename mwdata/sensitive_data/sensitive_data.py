from functools import reduce

from presidio_analyzer import AnalyzerEngine
import pandas as pd

engine = AnalyzerEngine(default_score_threshold=0.5, enable_trace_pii=True)


def sensitive_data(
    df,
    redact=True,
    anonymize=False,
    identify_column_infotypes=False,
    score_threshold=0.2,
    sample_size=100,
    cols=[],
):
    """Identifies, redacts, and anonymizes PII data

    Args:
        df: The dataframe
        redact: If True, redact sensitive data
        anonymize: If True, anonymize data
        identify_column_infotypes: If True identifies infotypes for each column
        score_threshold: Minimum confidence value for detected entities to be returned
        cols: List of columns

    Returns:
        A dataframe if redact or anonymize is True.
        Dictionary of column infotypes if identify_column_infotypes is True
    """
    if not isinstance(df, pd.DataFrame):
        raise NotImplementedError("Pandas data frame required")
    if not isinstance(cols, list):
        raise NotImplementedError("cols must be type list")
    if cols:
        df = df[cols]
    if redact is True:
        return redact_df(df, score_threshold)
    if identify_column_infotypes is True:
        return identify_infotypes(df, sample_size)
    if anonymize is True:
        pass


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


def create_mapping(text, score_threshold=0.2):
    """Identifies sensitive data and creates a mapping of the hashed data

    Args:
        text: String value
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        word_mapping: Mapping of the hashed data with the redacted string
        text: String with hashed values
    """
    ref_text = text
    response = identify_pii(text, score_threshold)
    word_mapping = {}
    for r in response:
        hash_v = str(hash(text[r.start : r.end]))
        word_mapping[hash_v] = str("<" + r.entity_type + ">")
        ref_text = ref_text.replace(text[r.start : r.end], hash_v)
    text = ref_text
    return word_mapping, text


def redact_info(text, score_threshold=0.2):
    """Redact sensitive data using mapping between hashed values and infotype

    Args:
        text: A string
        score_threshold: Minimum confidence value for detected entities to be returned

    Returns:
        String with redacted information
    """
    word_mapping, text = create_mapping(text, score_threshold)
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
