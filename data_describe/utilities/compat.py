try:
    import geopandas  # noqa: F401
    import shapely  # noqa: F401
    import geoplot  # noqa: F401
    import descartes  # noqa: F401

    _GEO_INSTALLED = True
except ImportError:
    _GEO_INSTALLED = False

try:
    import nltk  # noqa: F401
    import pyldavis  # noqa: F401
    import gensim  # noqa: F401

    _NLP_INSTALLED = True
except ImportError:
    _NLP_INSTALLED = False

try:
    import gcsfs  # noqa: F401

    _GCP_INSTALLED = True
except ImportError:
    _GCP_INSTALLED = False

try:
    import xlrd  # noqa: F401

    _EXCEL_INSTALLED = True
except ImportError:
    _EXCEL_INSTALLED = False
