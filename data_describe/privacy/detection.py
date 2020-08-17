from typing import Optional
import warnings

from data_describe.backends import _get_compute_backend
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.config._config import get_option
from data_describe._widget import BaseWidget
from data_describe.privacy.engine import engine


_DEFAULT_SCORE_THRESHOLD = get_option("sensitive_data.score_threshold")
_SAMPLE_SIZE = get_option("sensitive_data.sample_size")


def sensitive_data(
    df,
    mode: str = "redact",
    detect_infotypes: bool = True,
    columns: Optional[list] = None,
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
    sample_size: int = _SAMPLE_SIZE,
    engine_backend=engine,
    compute_backend: Optional[str] = None,
):
    """Identifies, redacts, and encrypts PII data.

    Note: sensitive_data uses Microsoft's Presidio in the backend. Presidio can be help identify sensitive data.
    However, because Presidio uses trained ML models, there is no guarantee that Presidio will find all sensitive information.

    Args:
        df (DataFrame): The dataframe

        mode (str): Select 'redact' or 'encrypt'.
            redact: Redact the sensitive data
            encrypt: Anonymize the sensitive data

        detect_infotypes (bool): If True, identifies infotypes for each column
        columns ([str]): Defaults to None
        score_threshold (float): Minimum confidence value for detected entities to be returned. Default is 0.2.
        sample_size (int): Number of sampled rows used for identifying column infotypes. Default is 100.
        engine_backend: The backend analyzer engine. Default is presidio_analyzer.
        compute_backend (str): Select compute backend

    Returns:
        SensitiveDataWidget
    """
    if not isinstance(df, _DATAFRAME_TYPE):
        raise TypeError("Pandas data frame or modin data frame required")

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
