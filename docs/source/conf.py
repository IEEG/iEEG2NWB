# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../../ieeg2nwb'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IEEG2NWB'
copyright = '2024, Noah Markowitz, Stephan Bickel'
author = 'Noah Markowitz, Stephan Bickel'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx.ext.viewcode']
              # Temporarily disabled Sphinx-Gallery
              # 'sphinx_gallery.gen_gallery']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Sphinx Gallery options -------------------------------------------------
# Temporarily disabled
# sphinx_gallery_conf = {
#     'examples_dirs': '../../examples',
#     'gallery_dirs': 'auto_examples',
#     'filename_pattern': '.py',
#     'ignore_pattern': r'__init__\.py',
#     'download_all_examples': False,
#     'thumbnail_size': (400, 400),
#     'capture_repr': (),
#     'matplotlib_animations': False,
# }

# -- Autodoc options -------------------------------------------------------
# Skip problematic modules temporarily
autodoc_mock_imports = ['ndx_events', 'colorama']

# -- Autosummary options ---------------------------------------------------
autosummary_generate = True
