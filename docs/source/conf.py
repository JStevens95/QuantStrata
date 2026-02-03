# QuantStrata Sphinx configuration
from pathlib import Path
import sys

# Add project root so that 'src' can be imported
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

project = "QuantStrata"
copyright = "QuantStrata"
author = "QuantStrata"
release = "0.1"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_use_param = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
