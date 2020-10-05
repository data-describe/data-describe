import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

project = "data-describe"
copyright = "2020, Maven Wave Partners"  # noqa: A001
author = "Maven Wave Atos Data Science"

extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "nbsphinx",
    "sphinx_multiversion",
]

# API Generation
autoapi_dirs = ["../../data_describe"]
autoapi_root = "."
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_add_toctree_entry = False
autoapi_template_dir = "_autoapi_templates"
autoapi_keep_files = True

# Multiversioning
smv_remote_whitelist = r"^.*$"
smv_branch_whitelist = "master"
html_sidebars = {"**": ["versioning.html"]}

# Theme / Style
templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.png"
html_theme_options = {"logo_only": True}
html_css_files = ["css/style.css"]
