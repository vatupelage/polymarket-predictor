# live_trader/db_model.py
"""Loads the trained dollar-bar PTB model bundle and returns a calibrated P(up).

The bundle (saved by train/train_db_model.py) is a dict:
  {"model": fitted estimator, "calibrator": isotonic/None,
   "feature_names": [...], "threshold_usd": float, "meta": {...}}
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np

# We predict from a plain ndarray (features built in a fixed order), but the
# estimators were fit on a named-column DataFrame. The order matches, so the
# "X does not have valid feature names" warning is noise — silence it so it
# doesn't spam the live bot log on every window.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


class DbModel:
    def __init__(self, path: str):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.calibrator = bundle.get("calibrator")
        self.feature_names = bundle["feature_names"]
        self.threshold_usd = bundle["threshold_usd"]
        self.meta = bundle.get("meta", {})
        # Window length + decision instant (secs-to-close). New bundles store these
        # at top level; older BTC-5m bundles predate them, so fall back to the
        # 5m config the bot has always used (300s window, decide at s2c=240).
        self.window_s = int(bundle.get("window_s", self.meta.get("window_s", 300)))
        self.monitor_start_s = int(
            bundle.get("monitor_start_s", self.meta.get("monitor_start_s", 240)))

    def predict_detailed(self, features: dict) -> dict:
        """Return both the pre-calibration and calibrated P(up). Lets the paper
        log capture whether isotonic calibration is helping or hurting."""
        x = np.array([[features[name] for name in self.feature_names]], dtype=float)
        raw = float(self.model.predict_proba(x)[0, 1])
        p_up = raw
        if self.calibrator is not None:
            p_up = float(self.calibrator.predict([raw])[0])
        return {"p_up": min(max(p_up, 0.0), 1.0), "raw": raw}

    def predict_p_up(self, features: dict) -> float:
        return self.predict_detailed(features)["p_up"]
