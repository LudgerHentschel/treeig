from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .lightgbm_backend import (
    _extract_lightgbm_booster,
    _lightgbm_model_kind,
    _predict_lightgbm_model,
)
from .sklearn_backend import (
    _extract_sklearn_decision_tree_regressor,
    _extract_sklearn_forest_regressor,
    _extract_sklearn_gradient_boosting_classifier,
    _extract_sklearn_gradient_boosting_regressor,
    _predict_sklearn_model,
)
from .xgboost_backend import (
    _extract_xgboost_booster,
    _predict_xgboost_model,
    _xgboost_model_kind,
)


def extract_tree_arrays(model: Any, target: Optional[int] = None) -> Dict[str, Any]:
    """Extract supported tree models into the common TreeIG internal format."""
    try:
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.tree import DecisionTreeRegressor
    except ImportError:
        GradientBoostingRegressor = None
        RandomForestRegressor = None
        ExtraTreesRegressor = None
        GradientBoostingClassifier = None
        DecisionTreeRegressor = None

    if GradientBoostingRegressor is not None and isinstance(model, GradientBoostingRegressor):
        if target not in (None, 0):
            raise ValueError("Regression target must be None or 0.")
        return _extract_sklearn_gradient_boosting_regressor(model)

    if DecisionTreeRegressor is not None and isinstance(model, DecisionTreeRegressor):
        if target not in (None, 0):
            raise ValueError("Regression target must be None or 0.")
        return _extract_sklearn_decision_tree_regressor(model)

    if RandomForestRegressor is not None and isinstance(model, RandomForestRegressor):
        if target not in (None, 0):
            raise ValueError("Regression target must be None or 0.")
        return _extract_sklearn_forest_regressor(
            model,
            backend="sklearn_random_forest_regressor",
        )

    if ExtraTreesRegressor is not None and isinstance(model, ExtraTreesRegressor):
        if target not in (None, 0):
            raise ValueError("Regression target must be None or 0.")
        return _extract_sklearn_forest_regressor(
            model,
            backend="sklearn_extra_trees_regressor",
        )

    if GradientBoostingClassifier is not None and isinstance(model, GradientBoostingClassifier):
        return _extract_sklearn_gradient_boosting_classifier(model, target)

    if _xgboost_model_kind(model) is not None:
        return _extract_xgboost_booster(model, target)

    if _lightgbm_model_kind(model) is not None:
        return _extract_lightgbm_booster(model, target)

    raise TypeError(
        "Unsupported model type. TreeIG currently supports selected sklearn, "
        "XGBoost, and LightGBM regression/additive-score classification models. "
        f"Got {type(model).__name__}."
    )


def model_predict(model: Any, X: np.ndarray, target: Optional[int] = None) -> np.ndarray:
    """Predict the scalar output being attributed."""
    if _xgboost_model_kind(model) is not None:
        return _predict_xgboost_model(model, X, target)

    if _lightgbm_model_kind(model) is not None:
        return _predict_lightgbm_model(model, X, target)

    return _predict_sklearn_model(model, X, target)
