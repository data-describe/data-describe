from functools import wraps

_DEPENDENCY_INSTALLED = {}


def requires(package_name):
    def f(func):
        @wraps(func)
        def g(*args, **kwargs):
            if not _DEPENDENCY_INSTALLED[package_name]:
                raise ImportError(
                    f"Package {package_name} required to use this function"
                )
            return func(*args, **kwargs)

        return g

    return f


try:
    import nltk  # noqa: F401

    _DEPENDENCY_INSTALLED["nltk"] = True
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("stem/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("stopwords")
    except LookupError:
        nltk.download("stopwords")
except ImportError:
    _DEPENDENCY_INSTALLED["nltk"] = False

try:
    import pyldavis  # noqa: F401

    _DEPENDENCY_INSTALLED["pyldavis"] = True
except ImportError:
    _DEPENDENCY_INSTALLED["pyldavis"] = False

try:
    import gensim  # noqa: F401

    _DEPENDENCY_INSTALLED["gensim"] = True
except ImportError:
    _DEPENDENCY_INSTALLED["gensim"] = False

try:
    import gcsfs  # noqa: F401

    _DEPENDENCY_INSTALLED["gcsfs"] = True
except ImportError:
    _DEPENDENCY_INSTALLED["gcsfs"] = False

try:
    import google.cloud.storage  # noqa: F401

    _DEPENDENCY_INSTALLED["google-cloud-storage"] = True
except ImportError:
    _DEPENDENCY_INSTALLED["google-cloud-storage"] = False

try:
    import modin.pandas
    import pandas

    _DEPENDENCY_INSTALLED["modin"] = True
    _FRAME_TYPE = (pandas.DataFrame, modin.pandas.DataFrame)
except ImportError:
    _DEPENDENCY_INSTALLED["modin"] = False
    _FRAME_TYPE = pandas.DataFrame
