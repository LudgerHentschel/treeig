# Changelog

## 0.2.0

- Add optional `treeig.GPUTreeIG` prediction attribution with persistent CUDA model and weighted-baseline state and reusable observation buffers.
- Keep CPU `TreeIG` as the default; load CUDA only when constructing `GPUTreeIG`.
- Add the `cuda` installation extra, simulator equivalence tests, and GPU usage and benchmark documentation.
- Correct GitHub project links and restrict release publishing to version tags.

GPUTreeIG is part of `treeig`, not a separate distribution. GPU performance depends on the workload; existing T4 measurements are examples rather than guarantees.
