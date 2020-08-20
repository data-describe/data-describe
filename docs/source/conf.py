import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

project = "data-describe"
doc_copyright = "2020 Maven Wave Partners"
author = "Maven Wave Data Science"

extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.napoleon",
    "sphinx.ext.inheritance_diagram",
    "autoapi.sphinx",
    "nbsphinx",
    "sphinx_multiversion",
]

autoapi_modules = {"data_describe": {"prune": True, "output": "_api"}}

templates_path = ["_templates"]

html_sidebars = {"**": ["versioning.html"]}

# Multiversioning
smv_branch_whitelist = "master"

exclude_patterns = []

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_css_files = [
    "css/style.css",
]
