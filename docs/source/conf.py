import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

project = "data-describe"
doc_copyright = "2020 Maven Wave Partners"
author = "Maven Wave Data Science"

extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "nbsphinx",
    "sphinx_multiversion",
]

autoapi_dirs = ['../../data_describe']

templates_path = ["_templates"]

html_sidebars = {"**": ["versioning.html"]}

# Multiversioning
smv_branch_whitelist = "feature/apidoc.*"

exclude_patterns = []

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_css_files = [
    "css/style.css",
]
