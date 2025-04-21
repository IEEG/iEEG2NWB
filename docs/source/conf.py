# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IEEG2NWB'
copyright = '2025, Noah Markowitz, Stephan Bickel'
author = 'Noah Markowitz, Stephan Bickel'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_gallery.gen_gallery',
    'nbsphinx',
]

# Optional nbsphinx configuration
nbsphinx_execute = 'never'  # Don't run notebooks during build

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Configure sphinx-gallery
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',   # path to example scripts
    'gallery_dirs': 'auto_examples',  # where to save gallery generated output
    'filename_pattern': '/example_',  # Only run files starting with "example_"
    'line_numbers': True,             # Show line numbers in code blocks
}

# Choose a theme (ReadTheDocs theme is popular)
html_theme = 'sphinx_rtd_theme'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
