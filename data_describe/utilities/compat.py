from functools import wraps
from typing import Dict


_PACKAGE_INSTALLED: Dict[str, bool] = {}


def requires(package_name):
    def f(func):
        @wraps(func)
        def g(*args, **kwargs):
            if not _PACKAGE_INSTALLED[package_name]:
                raise ModuleNotFoundError(
                    f"Package {package_name} required to use this function"
                )
            return func(*args, **kwargs)

        return g

    return f


try:
    import nltk  # noqa: F401

    _PACKAGE_INSTALLED["nltk"] = True
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
except (ImportError, ModuleNotFoundError):
    _PACKAGE_INSTALLED["nltk"] = False

try:
    import pyldavis  # noqa: F401

    _PACKAGE_INSTALLED["pyldavis"] = True
except (ImportError, ModuleNotFoundError):
    _PACKAGE_INSTALLED["pyldavis"] = False

try:
    import gensim  # noqa: F401

    _PACKAGE_INSTALLED["gensim"] = True
except (ImportError, ModuleNotFoundError):
    _PACKAGE_INSTALLED["gensim"] = False

try:
    import gcsfs  # noqa: F401

    _PACKAGE_INSTALLED["gcsfs"] = True
except (ImportError, ModuleNotFoundError):
    _PACKAGE_INSTALLED["gcsfs"] = False

try:
    import google.cloud.storage  # noqa: F401

    _PACKAGE_INSTALLED["google-cloud-storage"] = True
except (ImportError, ModuleNotFoundError):
    _PACKAGE_INSTALLED["google-cloud-storage"] = False

try:
    import modin.pandas
    import pandas

    _PACKAGE_INSTALLED["modin"] = True
except (ImportError, ModuleNotFoundError):
    import pandas

    _PACKAGE_INSTALLED["modin"] = False
