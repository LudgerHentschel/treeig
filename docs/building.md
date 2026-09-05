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
artifact. On pushes to `main` (or a manual run on `main`), it also publishes
the HTML to [GitHub Pages](https://ludgerhentschel.github.io/treeig/).
Pull requests build the documentation without deploying it.

For initial setup, open the repository's **Settings → Pages** and select
**GitHub Actions** as the build and deployment source. Then push the workflow
or run **Documentation** manually from the Actions tab. A successful `deploy`
job publishes the site. No package version change or release tag is required.

Edit the topic pages under `docs/`; keep the README focused on installation,
a first example, and links into the guide. Version information comes from
`pyproject.toml`. Generated HTML should not be committed.
