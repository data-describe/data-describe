..

Installation
============================================
data-describe can be installed using pip:

.. code-block:: bash

   pip install data_describe

data-describe also allows for optional functionality for advanced analyses (for example, text processing). These additional dependencies are not installed by default. To install these optional features, use the square bracket notation for installing these extras_: ::

   pip install data_describe[nlp]

Currently available "extras" options:

- all: Installs all extras
- nlp: Text / Natural Language Processing support
- gcp: Google Cloud support
- pii: Sensitive data support using presidio
- modin: Large dataset support using modin

.. _extras: https://packaging.python.org/tutorials/installing-packages/#installing-setuptools-extras