import data_describe

project = "data-describe"
doc_copyright = "2020 Maven Wave Partners"
author = "Maven Wave Data Science"

release = data_describe.__version__

extensions = ["sphinx.ext.napoleon", "nbsphinx", "sphinx_multiversion"]

templates_path = ["_templates"]

html_sidebars = {"**": ["versioning.html"]}

# Multiversioning
smv_branch_whitelist = "master"

exclude_patterns = []

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]
