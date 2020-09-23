import os
from typing import Dict

from setuptools import find_packages, setup

# Package meta-data.
NAME = "data_describe"
DESCRIPTION = "data-describe"
URL = "https://github.com/data-describe/data-describe"
EMAIL = ""
AUTHOR = "https://github.com/data-describe/data-describe/graphs/contributors"
REQUIRES_PYTHON = ">=3.7"
version: Dict[str, str] = {}
with open(os.path.join("data_describe", "_version.py")) as fp:
    exec(fp.read(), version)

REQUIRED = [
    "pandas>=1.0",
    "numpy>=1.16",
    "scipy>=1.1",
    "scikit-learn>=0.23",
    "seaborn>=0.11",
    "plotly>=4.0",
]

EXTRAS = {
    "nlp": ["nltk>=3.4", "pyldavis>=2.1.2", "gensim>=3.4.0"],
    "gcp": ["gcsfs>=0.2.1", "google-cloud-storage>=1.18.0"],
    "pii": ["presidio-analyzer==0.3.8917rc0"],
    "modin": ["modin>=0.7.3", "ray>=0.8.4"],
    "cluster": ["hdbscan>=0.8.17"],
    "time": ["statsmodels>=0.10"],
    "scatter": ["pyscagnostics>=0.1.0a4"],
    "all": [
        "nltk>=3.4",
        "pyldavis>=2.1.2",
        "gensim>=3.4.0",
        "gcsfs>=0.2.1",
        "google-cloud-storage>=1.18.0",
        "presidio-analyzer==0.3.8917rc0",
        "modin>=0.7.3",
        "ray>=0.8.4",
        "statsmodels>=0.10",
        "hdbscan>=0.8.17",
        "pyscagnostics>=0.1.0a4",
    ],
}


# Where the magic happens:
setup(
    name=NAME,
    version=version["__version__"],
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    entry_points={
        "data_describe_viz_backends": [
            "seaborn = data_describe.backends.viz:_seaborn",
            "plotly = data_describe.backends.viz:_plotly",
            "pyLDAvis = data_describe.backends.viz:_pyLDAvis",
        ],
        "data_describe_compute_backends": [
            "pandas = data_describe.backends.compute:_pandas",
            "modin = data_describe.backends.compute:_modin",
        ],
    },
)
