"""Exact CUDA backend for specialized, large-scale prediction attribution.

Most users should access this implementation through :class:`GPUTreeIG`. The
lower-level stateless entry point remains useful for benchmarks and validation.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

try:
    from numba import cuda
except ImportError as exc:
    raise ImportError(
        "GPUTreeIG requires CUDA support. Install treeig[cuda] on an NVIDIA "
        "CUDA host, or use TreeIG for CPU attribution."
    ) from exc


MAX_SCRATCH_WIDTH = 1024


def _cuda_module():
    return cuda


def cuda_available() -> bool:
    """Return whether a CUDA device or the CUDA simulator is available."""
    try:
        return bool(_cuda_module().is_available())
    except Exception:
        return False


def _power_of_two_width(required: int, name: str) -> int:
    required = int(required)
    if required < 1:
        raise ValueError(f"{name} scratch requires a positive capacity.")
    width = 1
    while width < required:
        width *= 2
    if width > MAX_SCRATCH_WIDTH:
        raise ValueError(
            f"CUDA backend requires {name} scratch width <= "
            f"{MAX_SCRATCH_WIDTH}; got required capacity {required}."
        )
    return width


def _scratch_widths(children_left, children_right):
    """Return exact power-of-two bounds for DFS stack and leaf segments.

    A binary-tree DFS needs at most ``max_depth + 1`` pending entries. Each
    leaf has a unique root path and can emit at most one path interval, so the
    segment buffer needs at most the maximum leaf count of any packed tree.
    Padding nodes are ignored because traversal starts at each root.
    """
    cl = np.asarray(children_left)
    cr = np.asarray(children_right)
    if cl.ndim != 2 or cr.shape != cl.shape or cl.shape[1] == 0:
        raise ValueError("children arrays must be aligned nonempty 2-D arrays.")

    maximum_stack = 1
    maximum_leaves = 1
    node_capacity = cl.shape[1]
    for tree in range(cl.shape[0]):
        pending = [(0, 0)]
        leaves = 0
        max_depth = 0
        while pending:
            node, depth = pending.pop()
            if node < 0 or node >= node_capacity:
                raise ValueError("Tree child index is outside the packed array.")
            left = int(cl[tree, node])
            right = int(cr[tree, node])
            max_depth = max(max_depth, depth)
            if left == right:
                leaves += 1
                continue
            if left < 0 or right < 0:
                raise ValueError("Internal tree nodes must have two valid children.")
            pending.append((right, depth + 1))
            pending.append((left, depth + 1))
        maximum_stack = max(maximum_stack, max_depth + 1)
        maximum_leaves = max(maximum_leaves, leaves)

    return (
        _power_of_two_width(maximum_stack, "stack"),
        _power_of_two_width(maximum_leaves, "segment"),
    )


@lru_cache(maxsize=None)
def _kernel_for_widths(stack_width: int, segment_width: int):
    from numba import float32, float64, int64

    @cuda.jit(device=True, inline=True)
    def go_left(xj, threshold, left_inclusive, round_input):
        if round_input:
            xj = float64(float32(xj))
        if left_inclusive:
            return xj <= threshold
        return xj < threshold

    @cuda.jit(device=True, inline=True)
    def path_value(x0, x1, j, t):
        if t <= 0.0:
            return x0[j]
        if t >= 1.0:
            return x1[j]
        return x0[j] + t * (x1[j] - x0[j])

    @cuda.jit(device=True)
    def predict_leaf_at_time(
        cl, cr, feat, threshold, value, left_inclusive, round_input,
        tree, x0, x1, t,
    ):
        node = 0
        while cl[tree, node] != cr[tree, node]:
            j = feat[tree, node]
            xj = path_value(x0, x1, j, t)
            if go_left(
                xj, threshold[tree, node], left_inclusive[tree, node],
                round_input[tree, node],
            ):
                node = cl[tree, node]
            else:
                node = cr[tree, node]
        return value[tree, node]

    @cuda.jit(device=True)
    def divergent_feature_at_times(
        cl, cr, feat, threshold, left_inclusive, round_input,
        tree, x0, x1, t_before, t_after,
    ):
        node_before = 0
        node_after = 0
        parent = -1
        while True:
            if node_before != node_after:
                if parent >= 0:
                    return feat[tree, parent]
                return -1
            left = cl[tree, node_before]
            right = cr[tree, node_before]
            if left == right:
                return -1
            parent = node_before
            j = feat[tree, node_before]
            c = threshold[tree, node_before]
            inclusive = left_inclusive[tree, node_before]
            rounded = round_input[tree, node_before]
            before = path_value(x0, x1, j, t_before)
            after = path_value(x0, x1, j, t_after)
            node_before = left if go_left(before, c, inclusive, rounded) else right
            node_after = left if go_left(after, c, inclusive, rounded) else right

    @cuda.jit
    def attribute_kernel(
        cl, cr, feat, threshold, value, left_inclusive, round_input,
        baselines, baseline_weights, X, y0, tree_weight, time_tol,
        phis, event_counts,
    ):
        q = cuda.grid(1)
        n_trees = cl.shape[0]
        n_baselines = baselines.shape[0]
        n_tasks = X.shape[0] * n_baselines * n_trees
        if q >= n_tasks:
            return

        tree = q % n_trees
        pair = q // n_trees
        baseline_index = pair % n_baselines
        observation = pair // n_baselines
        baseline_weight = baseline_weights[baseline_index]
        if baseline_weight == 0.0:
            return

        x0 = baselines[baseline_index]
        x1 = X[observation]
        stack_node = cuda.local.array(stack_width, int64)
        stack_tl = cuda.local.array(stack_width, float64)
        stack_tr = cuda.local.array(stack_width, float64)
        stack_feature = cuda.local.array(stack_width, int64)
        segment_tl = cuda.local.array(segment_width, float64)
        segment_value = cuda.local.array(segment_width, float64)
        segment_feature = cuda.local.array(segment_width, int64)

        stack_node[0] = 0
        stack_tl[0] = 0.0
        stack_tr[0] = 1.0
        stack_feature[0] = -1
        n_stack = 1
        n_segments = 0

        while n_stack > 0:
            n_stack -= 1
            node = stack_node[n_stack]
            tl = stack_tl[n_stack]
            tr = stack_tr[n_stack]
            event_feature = stack_feature[n_stack]
            left = cl[tree, node]
            right = cr[tree, node]

            if left == right:
                segment_tl[n_segments] = tl
                segment_value[n_segments] = value[tree, node]
                segment_feature[n_segments] = event_feature
                n_segments += 1
                continue

            j = feat[tree, node]
            c = threshold[tree, node]
            d = x1[j] - x0[j]
            inclusive = left_inclusive[tree, node]
            rounded = round_input[tree, node]

            if abs(d) <= time_tol * (abs(x0[j]) + abs(x1[j]) + 1.0):
                child = left if go_left(x0[j], c, inclusive, rounded) else right
                stack_node[n_stack] = child
                stack_tl[n_stack] = tl
                stack_tr[n_stack] = tr
                stack_feature[n_stack] = event_feature
                n_stack += 1
                continue

            crossing = (c - x0[j]) / d
            if tl + time_tol < crossing < tr - time_tol:
                before = left if d > 0.0 else right
                after = right if d > 0.0 else left
                stack_node[n_stack] = after
                stack_tl[n_stack] = crossing
                stack_tr[n_stack] = tr
                stack_feature[n_stack] = j
                n_stack += 1
                stack_node[n_stack] = before
                stack_tl[n_stack] = tl
                stack_tr[n_stack] = crossing
                stack_feature[n_stack] = event_feature
                n_stack += 1
            else:
                midpoint = 0.5 * (tl + tr)
                x_mid = x0[j] + midpoint * d
                child = left if go_left(x_mid, c, inclusive, rounded) else right
                stack_node[n_stack] = child
                stack_tl[n_stack] = tl
                stack_tr[n_stack] = tr
                stack_feature[n_stack] = event_feature
                n_stack += 1

        for a in range(1, n_segments):
            key_time = segment_tl[a]
            key_value = segment_value[a]
            key_feature = segment_feature[a]
            b = a - 1
            while b >= 0 and segment_tl[b] > key_time:
                segment_tl[b + 1] = segment_tl[b]
                segment_value[b + 1] = segment_value[b]
                segment_feature[b + 1] = segment_feature[b]
                b -= 1
            segment_tl[b + 1] = key_time
            segment_value[b + 1] = key_value
            segment_feature[b + 1] = key_feature

        task_weight = baseline_weight * tree_weight[tree]
        count = 0
        for k in range(1, n_segments):
            j = segment_feature[k]
            jump = segment_value[k] - segment_value[k - 1]
            if j >= 0 and jump != 0.0:
                cuda.atomic.add(phis, (observation, j), task_weight * jump)
                count += 1

        jump0 = 0.0
        if n_segments > 0:
            jump0 = segment_value[0] - y0[baseline_index, tree]
        if jump0 != 0.0:
            probe_time = 0.5
            if n_segments > 1:
                probe_time = time_tol + 0.5 * (
                    segment_tl[1] - time_tol
                )
            j0 = divergent_feature_at_times(
                cl, cr, feat, threshold, left_inclusive, round_input,
                tree, x0, x1, 0.0, probe_time,
            )
            if j0 >= 0:
                cuda.atomic.add(phis, (observation, j0), task_weight * jump0)
                count += 1

        y1 = predict_leaf_at_time(
            cl, cr, feat, threshold, value, left_inclusive, round_input,
            tree, x0, x1, 1.0,
        )
        previous = y1
        if n_segments > 0:
            previous = segment_value[n_segments - 1]
        jump1 = y1 - previous
        if jump1 != 0.0:
            probe_time = 0.5
            if n_segments > 0:
                probe_time = 0.5 * (
                    segment_tl[n_segments - 1] + (1.0 - time_tol)
                )
            j1 = divergent_feature_at_times(
                cl, cr, feat, threshold, left_inclusive, round_input,
                tree, x0, x1, probe_time, 1.0,
            )
            if j1 >= 0:
                cuda.atomic.add(phis, (observation, j1), task_weight * jump1)
                count += 1

        if count:
            cuda.atomic.add(
                event_counts, observation, baseline_weight * float64(count)
            )

    return attribute_kernel


def attribute_cuda(
    arrays,
    baselines,
    baseline_weights,
    X,
    time_tol,
    y0_per_baseline_tree,
    *,
    threads_per_block=128,
):
    """Compute exact weighted prediction attributions on a CUDA device.

    This prototype includes allocation and host/device transfers. It accepts
    TreeIG's internal packed arrays and prepared float64 inputs. Loss
    attribution and public ``TreeIG`` dispatch are deliberately out of scope.
    """
    cuda = _cuda_module()
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Set NUMBA_ENABLE_CUDASIM=1 before importing "
            "Numba for semantic testing without an NVIDIA GPU."
        )

    baselines = np.ascontiguousarray(baselines, dtype=np.float64)
    baseline_weights = np.ascontiguousarray(baseline_weights, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    y0 = np.ascontiguousarray(y0_per_baseline_tree, dtype=np.float64)
    n_obs, n_features = X.shape
    n_trees = arrays["children_left"].shape[0]
    if baselines.ndim != 2 or baselines.shape[1] != n_features:
        raise ValueError("baselines and X must be aligned 2-D arrays.")
    if baseline_weights.shape != (baselines.shape[0],):
        raise ValueError("baseline_weights must align with baseline rows.")
    if y0.shape != (baselines.shape[0], n_trees):
        raise ValueError("y0_per_baseline_tree has an incompatible shape.")

    scratch_widths = _scratch_widths(
        arrays["children_left"], arrays["children_right"]
    )
    kernel = _kernel_for_widths(*scratch_widths)
    host_arrays = (
        arrays["children_left"], arrays["children_right"], arrays["feature"],
        arrays["threshold"], arrays["value"], arrays["left_inclusive"],
        arrays.get(
            "round_input", np.ones_like(arrays["children_left"], dtype=np.bool_)
        ),
    )
    device_arrays = [cuda.to_device(np.ascontiguousarray(a)) for a in host_arrays]
    d_baselines = cuda.to_device(baselines)
    d_weights = cuda.to_device(baseline_weights)
    d_X = cuda.to_device(X)
    d_y0 = cuda.to_device(y0)
    d_tree_weight = cuda.to_device(
        np.ascontiguousarray(arrays["tree_weight"], dtype=np.float64)
    )
    d_phis = cuda.to_device(np.zeros((n_obs, n_features), dtype=np.float64))
    d_counts = cuda.to_device(np.zeros(n_obs, dtype=np.float64))

    tasks = n_obs * baselines.shape[0] * n_trees
    blocks = (tasks + threads_per_block - 1) // threads_per_block
    if tasks:
        kernel[blocks, threads_per_block](
            *device_arrays, d_baselines, d_weights, d_X, d_y0, d_tree_weight,
            float(time_tol), d_phis, d_counts,
        )
        cuda.synchronize()
    return d_phis.copy_to_host(), d_counts.copy_to_host()


class CUDAAttributor:
    """Persistent device-resident prediction-attribution engine."""

    def __init__(
        self,
        arrays,
        baselines,
        baseline_weights,
        time_tol,
        y0_per_baseline_tree,
        *,
        threads_per_block=128,
    ):
        cuda = _cuda_module()
        if not cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. GPUTreeIG requires an NVIDIA CUDA "
                "device and a working Numba CUDA installation."
            )
        if not isinstance(threads_per_block, (int, np.integer)) or not (
            1 <= int(threads_per_block) <= 1024
        ):
            raise ValueError("threads_per_block must be an integer from 1 to 1024.")

        baselines = np.ascontiguousarray(baselines, dtype=np.float64)
        weights = np.ascontiguousarray(baseline_weights, dtype=np.float64)
        y0 = np.ascontiguousarray(y0_per_baseline_tree, dtype=np.float64)
        n_trees = arrays["children_left"].shape[0]
        n_features = int(arrays["n_features"])
        if baselines.ndim != 2 or baselines.shape[1] != n_features:
            raise ValueError("baselines must align with the model's feature count.")
        if weights.shape != (baselines.shape[0],):
            raise ValueError("baseline_weights must align with baseline rows.")
        if y0.shape != (baselines.shape[0], n_trees):
            raise ValueError("y0_per_baseline_tree has an incompatible shape.")

        scratch_widths = _scratch_widths(
            arrays["children_left"], arrays["children_right"]
        )
        self._cuda = cuda
        self._kernel = _kernel_for_widths(*scratch_widths)
        self._scratch_widths = scratch_widths
        self._threads_per_block = int(threads_per_block)
        self._time_tol = float(time_tol)
        self._n_features = n_features
        self._n_trees = n_trees
        self._n_baselines = baselines.shape[0]
        host_arrays = (
            arrays["children_left"], arrays["children_right"], arrays["feature"],
            arrays["threshold"], arrays["value"], arrays["left_inclusive"],
            arrays.get(
                "round_input",
                np.ones_like(arrays["children_left"], dtype=np.bool_),
            ),
        )
        self._device_arrays = tuple(
            cuda.to_device(np.ascontiguousarray(a)) for a in host_arrays
        )
        self._d_baselines = cuda.to_device(baselines)
        self._d_weights = cuda.to_device(weights)
        self._d_y0 = cuda.to_device(y0)
        self._d_tree_weight = cuda.to_device(
            np.ascontiguousarray(arrays["tree_weight"], dtype=np.float64)
        )
        self._capacity = 0
        self._d_X = None
        self._d_phis = None
        self._d_counts = None

    @property
    def capacity(self):
        """Number of observation rows accommodated by current reusable buffers."""
        return self._capacity

    def _ensure_capacity(self, n_obs):
        if n_obs <= self._capacity:
            return
        capacity = max(n_obs, max(1, 2 * self._capacity))
        cuda = self._cuda
        self._d_X = cuda.device_array(
            (capacity, self._n_features), dtype=np.float64
        )
        self._d_phis = cuda.device_array(
            (capacity, self._n_features), dtype=np.float64
        )
        self._d_counts = cuda.device_array(capacity, dtype=np.float64)
        self._capacity = capacity

    def attribute(self, X):
        """Attribute a prepared host batch while reusing resident fixed state."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self._n_features:
            raise ValueError("X must align with the model's feature count.")
        n_obs = X.shape[0]
        if n_obs == 0:
            return (
                np.empty((0, self._n_features), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        self._ensure_capacity(n_obs)
        d_X = self._d_X[:n_obs]
        d_phis = self._d_phis[:n_obs]
        d_counts = self._d_counts[:n_obs]
        d_X.copy_to_device(X)
        d_phis.copy_to_device(np.zeros_like(X))
        d_counts.copy_to_device(np.zeros(n_obs, dtype=np.float64))

        tasks = n_obs * self._n_baselines * self._n_trees
        blocks = (tasks + self._threads_per_block - 1) // self._threads_per_block
        self._kernel[blocks, self._threads_per_block](
            *self._device_arrays,
            self._d_baselines,
            self._d_weights,
            d_X,
            self._d_y0,
            self._d_tree_weight,
            self._time_tol,
            d_phis,
            d_counts,
        )
        self._cuda.synchronize()
        return d_phis.copy_to_host(), d_counts.copy_to_host()
