from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
project = "TreeIG"
author = "Ludger Hentschel"
copyright = "2026, Ludger Hentschel"
release = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax"]
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3
exclude_patterns = ["_build"]
html_theme = "pydata_sphinx_theme"
html_title = f"TreeIG {release}"
html_theme_options = {"github_url": "https://github.com/LudgerHentschel/treeig", "show_prev_next": True}
html_static_path = []
napoleon_numpy_docstring = True
napoleon_google_docstring = False
