# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys


# -- Project information -----------------------------------------------------

project = "Data Describe"
doc_copyright = "2019 Maven Wave"
author = "Akanksha Jindal, David Law, Richard Truong-Chau, Rishi Sheth, Ross Claytor, Jihua Wang"

# The full version, including alpha/beta/rc tags
release = "0.0.1a"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon", "nbsphinx", "sphinx_multiversion"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

html_sidebars = {"**": ["versioning.html"]}

# Multiversioning
smv_branch_whitelist = "feature.*"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Currently doesn't work? https://github.com/spatialaudio/nbsphinx/issues/128
html_theme = "sphinx_rtd_theme"
# html_theme_path = ["_themes", ]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]
