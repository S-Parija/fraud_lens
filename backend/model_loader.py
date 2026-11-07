"""Compatibility loader for this notebook's trusted XGBoost pickle exports.

The Kaggle memory snapshots cannot be restored directly by this Windows build.
Each snapshot also contains a complete native UBJSON Model object. Load that
inference model instead of restoring the training-time Config/memory state.
The saved preprocessor, calibration, and thresholds are retained unchanged.
"""

import pickle
from pathlib import Path

from xgboost import Booster


class InferenceBooster(Booster):
    """Restore the native inference model from the supplied V3 snapshot format."""

    def __setstate__(self, state):
        raw = state["handle"]
        # UBJSON represents this key as L (int64 length), 5, then 'Model'.
        marker = b"L" + (5).to_bytes(8, "big") + b"Model"
        if not raw.startswith(b"{L") or raw.count(marker) != 1 or raw[-2:] != b"}}":
            raise ValueError("Unrecognized V3 booster snapshot. Re-export native XGBoost models from the notebook.")
        native_model = raw[raw.index(marker) + len(marker):-1]
        self.__init__()
        self.load_model(native_model)


class TrustedModelUnpickler(pickle.Unpickler):
    """Use the inference loader for Booster objects in this local trusted pickle.

    This is a compatibility adapter, not a sandbox for untrusted pickle files.
    """

    def find_class(self, module, name):
        if module == "xgboost.core" and name == "Booster":
            return InferenceBooster
        return super().find_class(module, name)


def load_portable_bundle(path: Path) -> dict:
    """Read the complete local bundle while adapting its embedded boosters."""
    with path.open("rb") as model_file:
        return TrustedModelUnpickler(model_file).load()
