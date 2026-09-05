"""Semantic tests for the isolated experimental CUDA backend."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np

from treeig.cuda_backend import _scratch_widths


def test_cuda_scratch_widths_use_depth_and_leaf_bounds():
    depth = 6
    node_count = 2 ** (depth + 1) - 1
    internal_count = 2**depth - 1
    left = -np.ones((1, node_count), dtype=np.int64)
    right = -np.ones((1, node_count), dtype=np.int64)
    for node in range(internal_count):
        left[0, node] = 2 * node + 1
        right[0, node] = 2 * node + 2

    assert _scratch_widths(left, right) == (8, 64)


def test_cuda_simulator_matches_cpu_weighted_endpoints():
    repository = Path(__file__).resolve().parents[1]
    program = textwrap.dedent(
        """
        import numpy as np
        from sklearn.tree import DecisionTreeRegressor

        from treeig import TreeIG
        from treeig.core import _compute_y0_per_baseline_tree
        from treeig.cuda_backend import attribute_cuda

        X = np.array([
            [0.0, -1.0], [0.25, 1.0], [0.75, -1.0], [1.0, 1.0],
            [0.1, 0.5], [0.9, -0.5],
        ], dtype=np.float64)
        y = np.array([0.0, 1.0, 2.0, 4.0, 0.5, 3.5])
        model = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
        threshold = float(model.tree_.threshold[0])
        baselines = np.array([
            [threshold, -1.0], [0.0, 1.0], [1.0, -1.0],
        ])
        weights = np.array([0.2, 0.0, 0.8])
        data = np.array([
            [threshold, 1.0], [0.0, -1.0], [1.0, 1.0],
        ])
        explainer = TreeIG(model)
        cpu = explainer.attribute(
            data, baseline=baselines, baseline_weights=weights
        )
        arrays = explainer._arrays
        y0 = _compute_y0_per_baseline_tree(arrays, baselines)
        gpu, _ = attribute_cuda(
            arrays, baselines, weights, data, explainer.time_tol, y0,
            threads_per_block=1,
        )
        np.testing.assert_allclose(gpu, cpu, atol=1e-12, rtol=1e-12)
        expected = model.predict(data) - weights @ model.predict(baselines)
        np.testing.assert_allclose(gpu.sum(axis=1), expected, atol=1e-12)
        """
    )
    environment = os.environ.copy()
    environment["NUMBA_ENABLE_CUDASIM"] = "1"
    environment["PYTHONPATH"] = str(repository)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_gpu_treeig_reuses_resident_state_across_calls_and_batches():
    repository = Path(__file__).resolve().parents[1]
    program = textwrap.dedent(
        """
        import numpy as np
        from sklearn.tree import DecisionTreeRegressor

        from treeig import GPUTreeIG, TreeIG

        train = np.array([
            [0.0, -1.0], [0.25, 1.0], [0.75, -1.0], [1.0, 1.0],
            [0.1, 0.5], [0.9, -0.5],
        ], dtype=np.float64)
        y = np.array([0.0, 1.0, 2.0, 4.0, 0.5, 3.5])
        model = DecisionTreeRegressor(max_depth=2, random_state=0).fit(train, y)
        threshold = float(model.tree_.threshold[0])
        baselines = np.array([
            [threshold, -1.0], [0.0, 1.0], [1.0, -1.0],
        ])
        weights = np.array([0.2, 0.0, 0.8])
        first = np.array([[threshold, 1.0]])
        second = np.array([
            [threshold, 1.0], [0.0, -1.0], [1.0, 1.0], [0.5, 0.0],
        ])

        cpu = TreeIG(model, baseline=baselines, baseline_weights=weights)
        gpu = GPUTreeIG(model, baseline=baselines, baseline_weights=weights,
                        threads_per_block=1)
        assert gpu.device_capacity == 0
        np.testing.assert_allclose(gpu.attribute(first), cpu.attribute(first),
                                   atol=1e-12, rtol=1e-12)
        first_capacity = gpu.device_capacity
        assert first_capacity >= 1
        np.testing.assert_allclose(gpu.attribute(second), cpu.attribute(second),
                                   atol=1e-12, rtol=1e-12)
        grown_capacity = gpu.device_capacity
        assert grown_capacity >= len(second) > first_capacity
        np.testing.assert_allclose(
            gpu.attribute(second, batch_size=2), cpu.attribute(second),
            atol=1e-12, rtol=1e-12,
        )
        explanation = gpu.explain(second)
        np.testing.assert_allclose(
            explanation.values, cpu.attribute(second), atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(explanation.completeness_error, 0.0, atol=1e-12)
        try:
            gpu.attribute(second, baseline=np.zeros_like(baselines))
        except ValueError as error:
            assert "fixed at construction" in str(error)
        else:
            raise AssertionError("GPUTreeIG accepted changed baseline state")
        try:
            gpu.loss_attribution(second, np.zeros(len(second)))
        except NotImplementedError as error:
            assert "attribute() and explain() only" in str(error)
        else:
            raise AssertionError("GPUTreeIG silently used CPU loss attribution")
        assert gpu.device_capacity == grown_capacity
        expected = model.predict(second) - weights @ model.predict(baselines)
        np.testing.assert_allclose(
            gpu.attribute(second).sum(axis=1), expected, atol=1e-12
        )

        rng = np.random.default_rng(23)
        deep_X = rng.normal(size=(256, 6))
        deep_y = (
            deep_X[:, 0] * deep_X[:, 1] - deep_X[:, 2] ** 2
            + np.sin(deep_X[:, 3]) + 0.2 * deep_X[:, 4]
        )
        deep_model = DecisionTreeRegressor(
            max_depth=6, random_state=4
        ).fit(deep_X, deep_y)
        deep_baselines = np.ascontiguousarray(deep_X[:8])
        deep_weights = np.arange(1.0, 9.0)
        deep_weights /= deep_weights.sum()
        deep_data = np.ascontiguousarray(deep_X[100:105])
        deep_cpu = TreeIG(
            deep_model, baseline=deep_baselines,
            baseline_weights=deep_weights,
        )
        deep_gpu = GPUTreeIG(
            deep_model, baseline=deep_baselines,
            baseline_weights=deep_weights, threads_per_block=1,
        )
        np.testing.assert_allclose(
            deep_gpu.attribute(deep_data), deep_cpu.attribute(deep_data),
            atol=1e-12, rtol=1e-12,
        )
        stack_width, segment_width = deep_gpu._gpu._scratch_widths
        assert stack_width <= 8
        assert segment_width <= 64
        """
    )
    environment = os.environ.copy()
    environment["NUMBA_ENABLE_CUDASIM"] = "1"
    environment["PYTHONPATH"] = str(repository)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
