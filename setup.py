#!/usr/bin/env python
from setuptools import find_packages, setup

# Package meta-data.
NAME = "data_describe"
DESCRIPTION = "Data Describe"
URL = "https://github.com/brianray/data-describe"
EMAIL = ""
AUTHOR = "https://github.com/brianray/data-describe/graphs/contributors"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = "0.2a"

REQUIRED = [
    "pandas>=0.24.0",
    "numpy>=1.15.4",
    "scipy>=1.1.0",
    "seaborn>=0.9.0",
    "scikit-learn>=0.20.3",
    "eli5>=0.8.1",
    "networkx>=2.2",
    "hdbscan>=0.8.20",
    "gcsfs>=0.2.1",
    "plotly>=3.8.1",
]

EXTRAS = {
    "nlp": ["nltk>=3.4", "pyldavis>=2.1.2", "gensim>=3.4.0"],
    "gcp": ["gcsfs>=0.2.1", "google-cloud-storage>=1.18.0"],
}


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
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
            "matplotlib = data_describe:backends.viz._matplotlib",
            "plotly = data_describe:backends.viz._plotly"
        ],
        "data_describe_compute_backends": [
            "pandas = data_describe:backends.compute._pandas",
            "modin = data_describe:backends.compute_modin",
        ]
    },
)
