"""Fail a release when its Git tag and package version disagree."""

import os
import pathlib
import sys

try:
    import tomllib
except ImportError:  # Python 3.9 and 3.10
    import tomli as tomllib


project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
version = project["project"]["version"]
if ".dev" in version or "+" in version:
    sys.exit("development and local versions must not be published")
expected_tag = f"v{version}"
actual_tag = os.environ.get("RELEASE_TAG") or os.environ.get("GITHUB_REF_NAME")

if actual_tag != expected_tag:
    sys.exit(f"release tag {actual_tag!r} does not match {expected_tag!r}")

print(f"release tag matches package version: {version}")
