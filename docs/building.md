# Building the documentation

The documentation uses Sphinx, MyST Markdown, and the PyData Sphinx Theme.
NumPy-style API docstrings are rendered with Sphinx's Napoleon extension.
No custom theme or frontend build is needed.

From the repository root, using Python 3.11 or later:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. Documentation dependencies are
optional and do not change TreeIG's runtime requirements. The documentation CI
builds HTML with warnings treated as errors and saves it as a downloadable
artifact. Publishing to a public documentation host is a separate step.

Edit the topic pages under `docs/`; keep the README focused on installation,
a first example, and links into the guide. Version information comes from
`pyproject.toml`. Generated HTML should not be committed.
