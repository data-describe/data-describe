from functools import wraps
from typing import Dict
import warnings

from IPython import get_ipython


_PACKAGE_INSTALLED: Dict[str, bool] = {}


def requires(package_name):
    """Import required packages."""

    def f(func):
        @wraps(func)
        def g(*args, **kwargs):
            if not _PACKAGE_INSTALLED[package_name]:
                raise ImportError(
                    f"Package {package_name} required to use this function"
                )
            return func(*args, **kwargs)

        return g

    return f


try:
    import nltk  # noqa: F401
    from nltk import word_tokenize  # noqa: F401
    from nltk.corpus import stopwords  # noqa: F401
    from nltk.stem import WordNetLemmatizer  # noqa: F401
    from nltk.stem.lancaster import LancasterStemmer  # noqa: F401
    from nltk.util import ngrams  # noqa: F401
    from nltk import FreqDist  # noqa: F401

    _PACKAGE_INSTALLED["nltk"] = True
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
except ImportError:
    _PACKAGE_INSTALLED["nltk"] = False

try:
    import pyLDAvis  # noqa: F401
    import pyLDAvis.gensim  # noqa: F401

    _PACKAGE_INSTALLED["pyLDAvis"] = True
except ImportError:
    _PACKAGE_INSTALLED["pyLDAvis"] = False

try:
    import gensim  # noqa: F401
    from gensim.corpora.dictionary import Dictionary  # noqa: F401
    from gensim.models.coherencemodel import CoherenceModel  # noqa: F401
    from gensim.models.ldamodel import LdaModel  # noqa: F401
    from gensim.models.lsimodel import LsiModel  # noqa: F401
    from gensim.summarization.summarizer import summarize  # noqa: F401

    _PACKAGE_INSTALLED["gensim"] = True
except ImportError:
    _PACKAGE_INSTALLED["gensim"] = False

try:
    import gcsfs  # noqa: F401

    _PACKAGE_INSTALLED["gcsfs"] = True
except ImportError:
    _PACKAGE_INSTALLED["gcsfs"] = False

try:
    from google.cloud import storage  # noqa: F401

    _PACKAGE_INSTALLED["google-cloud-storage"] = True
except ImportError:
    _PACKAGE_INSTALLED["google-cloud-storage"] = False

try:
    import spacy

    _PACKAGE_INSTALLED["spacy"] = True

    if not spacy.util.is_package("en_core_web_lg"):
        warnings.warn(
            "Downloading en_core_web_lg model for Spacy. This may take several minutes."
        )
        spacy.cli.download("en_core_web_lg")
except ImportError:
    _PACKAGE_INSTALLED["spacy"] = False


_DATAFRAME_BACKENDS = {
    "<class 'pandas.core.frame.DataFrame'>": ["pandas"],
    "<class 'pandas.core.series.Series'>": ["pandas"],
    "<class 'modin.pandas.dataframe.DataFrame'>": ["modin", "pandas"],
    "<class 'modin.pandas.series.Series'>": ["modin", "pandas"],
}

try:
    import modin.pandas
    import pandas

    _PACKAGE_INSTALLED["modin"] = True
    _SERIES_TYPE = (pandas.Series, modin.pandas.Series)
    _DATAFRAME_TYPE = (pandas.DataFrame, modin.pandas.DataFrame)
except ImportError:
    import pandas

    _PACKAGE_INSTALLED["modin"] = False
    _SERIES_TYPE = pandas.Series
    _DATAFRAME_TYPE = pandas.DataFrame


try:
    import hdbscan  # noqa: F401

    _PACKAGE_INSTALLED["hdbscan"] = True
except ImportError:
    _PACKAGE_INSTALLED["hdbscan"] = False

try:
    import presidio_analyzer  # noqa: F401

    _PACKAGE_INSTALLED["presidio_analyzer"] = True
except ImportError:
    _PACKAGE_INSTALLED["presidio_analyzer"] = False
_IN_NOTEBOOK = get_ipython() is not None
