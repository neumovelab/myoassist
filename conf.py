# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'MyoAssist'
copyright = '2025, MyoAssist Team'
author = 'MyoAssist Team'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# MyST Parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
] 