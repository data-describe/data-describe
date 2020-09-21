..

Installation
============================================
data-describe can be installed using pip:

.. code-block:: bash

   pip install data_describe

.. note::
   During internal beta-testing, data-describe must be installed from the source (GitHub). A source ZIP file should have been distributed to you; to install, run::

      pip install data-describe-master.zip # The name of the zip file, placed in your current directory

data-describe also allows for optional functionality for advanced analyses (for example, text processing). These additional dependencies are not installed by default. To install these optional features, use the square bracket notation for installing these extras_: ::

   pip install data_describe[nlp]

Currently available "extras" options:

- all: Installs all extras
- nlp: Text / Natural Language Processing support
- gcp: Google Cloud support
- pii: Sensitive data support using presidio
- modin: Large dataset support using modin

.. _extras: https://packaging.python.org/tutorials/installing-packages/#installing-setuptools-extras