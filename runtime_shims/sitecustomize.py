"""Runtime shims for fragile remote inference environments.

Python imports ``sitecustomize`` automatically on startup when it is present on
``PYTHONPATH``. We use that hook to provide a tiny sklearn.metrics fallback for
environments where sklearn/numpy are ABI-incompatible but transformers/vLLM
still try to import ``roc_curve``.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types


def install_sklearn_metrics_stub() -> None:
    try:
        from sklearn.metrics import roc_curve  # noqa: F401
        return
    except Exception:
        metrics_module = types.ModuleType("sklearn.metrics")
        metrics_module.__spec__ = importlib.machinery.ModuleSpec(
            "sklearn.metrics",
            loader=None,
        )

        def roc_curve(*args: object, **kwargs: object) -> None:
            raise RuntimeError("sklearn.metrics.roc_curve is unavailable in this runtime.")

        metrics_module.roc_curve = roc_curve

        sklearn_module = sys.modules.get("sklearn") or types.ModuleType("sklearn")
        sklearn_module.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
        sklearn_module.metrics = metrics_module

        sys.modules["sklearn"] = sklearn_module
        sys.modules["sklearn.metrics"] = metrics_module


install_sklearn_metrics_stub()
