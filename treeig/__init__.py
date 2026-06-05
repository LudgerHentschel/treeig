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

__all__ = [
    "TreeIG",
    "compute",
    "exact_gb_ig_batch_fast",
    "extract_gb_tree_arrays",
    "timed_call",
    "warmup_exact_gb_ig",
    "TreeIGNumeric",
    "compute_numeric",
]

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

__version__ = version("treeig")