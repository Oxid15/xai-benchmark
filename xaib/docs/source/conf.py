# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import xaib

# -- Project information -----------------------------------------------------

project = "xai-benchmark"
copyright = "2023, Ilia Moiseev"
author = "Ilia Moiseev"

# The full version, including alpha/beta/rc tags
release = xaib.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "sphinx_tags",
    "sphinx_design",
]

autodoc_default_options = {"undoc-members": False}
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_material"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


html_theme_options = {
    "base_url": "http://oxid15.github.io/xai-benchmark/",
    "repo_url": "https://github.com/Oxid15/xai-benchmark/",
    "repo_name": "xai-benchmark",
    "repo_type": "github",
    "globaltoc_depth": 2,
    "globaltoc_collapse": True,
    "color_primary": "white",
    "color_accent": "red",
    "html_minify": True,
    "css_minify": True,
    "nav_title": "XAIB - Open and extensible benchmark for XAI methods",
    "logo_icon": (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Dela+Gothic+One&display=swap"'
        ' rel="stylesheet"><ooo style="font-family: Dela Gothic One;font-size: 40px">細部</p>'
    ),  # "&#x8cea",  #'&#x7d30&#x90e8'
}

html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

# Tags configuration
tags_intro_text = ""
tags_page_title = "Tags"
tags_create_tags = True
tags_create_badges = True
tags_badge_colors = {
    "sk_dataset": "primary",
    "toy_dataset": "primary",
    "classification": "success",
    "white_box": "light",
    "black_box": "dark",
}
