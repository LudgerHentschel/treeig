from .api import (
    TreeIG,
    compute,
    exact_gb_ig_batch_fast,
    extract_gb_tree_arrays,
    timed_call,
    warmup_exact_gb_ig,
)
from .numeric import (
    TreeIGNumeric,
    compute_numeric,
)
from .explanation import Explanation
from .dispatch import supports_model as supports
from .gpu import GPUTreeIG

__all__ = [
    "TreeIG",
    "GPUTreeIG",
    "compute",
    "exact_gb_ig_batch_fast",
    "extract_gb_tree_arrays",
    "timed_call",
    "warmup_exact_gb_ig",
    "TreeIGNumeric",
    "compute_numeric",
    "supports",
    "Explanation",
]

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("treeig")
except PackageNotFoundError:
    __version__ = "0+unknown"
