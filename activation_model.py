"""
activation_model.py — inference wrapper around the WESAD-trained activation
classifiers (see the sibling analysis project's scripts/train_model.py for
how these were trained). Shared by local_app/ and web/ via processor.py,
exactly like specs/dyadic_biofeedback_metrics.py.

Two variants, both baseline-relative logistic regressions predicting
P(activated):
  "conversation" — hr_mean, hr_std, rmssd only. Used whenever coherence
                   isn't reliably available (i.e. most of an actual
                   conversation, where speech gates it off).
  "recovery"     — hr_mean, hr_std, rmssd, coherence. Used once a long
                   enough stationary window makes coherence trustworthy
                   (recovery/meditate mode).
"""

import warnings
from pathlib import Path

import joblib
import numpy as np

# The bundled StandardScaler was fit on a pandas DataFrame (training-side);
# feeding it a plain numpy array here (no pandas dependency in the app) is
# correct as long as column order matches design_columns, which _load()
# verifies — silence the resulting "no feature names" warning.
warnings.filterwarnings("ignore", message="X does not have valid feature names",
                         category=UserWarning)

MODEL_DIR = Path(__file__).parent / "model"

_FEATURE_ORDER = {
    "conversation": ("d_hr_mean", "d_hr_std", "d_log_rmssd"),
    "recovery": ("d_hr_mean", "d_hr_std", "d_log_rmssd", "d_log_coherence"),
}

_bundles: dict[str, dict] = {}


def _load(variant: str) -> dict:
    if variant not in _bundles:
        path = MODEL_DIR / f"activation_{variant}.joblib"
        bundle = joblib.load(path)
        if tuple(bundle["design_columns"]) != _FEATURE_ORDER[variant]:
            raise ValueError(
                f"{path} design_columns {bundle['design_columns']} != "
                f"expected {_FEATURE_ORDER[variant]}"
            )
        _bundles[variant] = bundle
    return _bundles[variant]


def predict_proba(variant: str, d_hr_mean: float, d_hr_std: float,
                   d_log_rmssd: float, d_log_coherence: float | None = None) -> float:
    """Return P(activated) in [0, 1] for the given variant."""
    bundle = _load(variant)
    values = {
        "d_hr_mean": d_hr_mean,
        "d_hr_std": d_hr_std,
        "d_log_rmssd": d_log_rmssd,
        "d_log_coherence": d_log_coherence,
    }
    x = np.array([[values[col] for col in bundle["design_columns"]]], dtype=float)
    x_scaled = bundle["scaler"].transform(x)
    return float(bundle["model"].predict_proba(x_scaled)[0, 1])
