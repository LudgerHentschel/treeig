"""CPU use must remain independent of the optional CUDA backend."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_cpu_attribution_without_cuda_imports():
    program = textwrap.dedent("""
        import importlib.abc
        import sys

        class BlockCUDA(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numba.cuda" or fullname.startswith("numba.cuda."):
                    raise ImportError("CUDA deliberately unavailable in this test")

        sys.meta_path.insert(0, BlockCUDA())
        import numpy as np
        from sklearn.tree import DecisionTreeRegressor
        from treeig import TreeIG, GPUTreeIG

        X = np.array([[0.0], [1.0]])
        model = DecisionTreeRegressor().fit(X, [0.0, 2.0])
        cpu = TreeIG(model, baseline=X[0])
        np.testing.assert_allclose(cpu.attribute(X), [[0.0], [2.0]])
        assert "treeig.cuda_backend" not in sys.modules
        assert "numba.cuda" not in sys.modules
        try:
            GPUTreeIG(model, baseline=X[0])
        except ImportError as error:
            assert "treeig[cuda]" in str(error)
        else:
            raise AssertionError("GPUTreeIG did not request CUDA")
    """)
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository)
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=repository, env=environment,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
