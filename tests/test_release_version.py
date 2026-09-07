"""The publishing guard must reject mismatches and development artifacts."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_release_version.py"


@pytest.mark.parametrize(
    "version,tag,success",
    [("0.1.1", "v0.1.1", True), ("0.1.1", "v0.1.2", False),
     ("0.1.1.dev1", "v0.1.1.dev1", False), ("0.1.1+local", "v0.1.1+local", False),
     ("0.1.1rc1", "v0.1.1rc1", True)],
)
def test_release_version_guard(tmp_path, version, tag, success):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    env = dict(os.environ, RELEASE_TAG=tag, GITHUB_REF_NAME="irrelevant")
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is success, result.stdout + result.stderr
