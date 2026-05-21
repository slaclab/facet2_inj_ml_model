"""FACET Model Package."""

import sys
from importlib import import_module

from facet2_inj_ml_model.loader import load_model

_LEGACY_TRAIN_EXPORTS = {
	"CovarianceAwareLoss",
	"CovarianceSurrogateModel",
	"build_model",
	"chol_vectors_to_covariance",
}


def __getattr__(name):
	if name in _LEGACY_TRAIN_EXPORTS:
		training_module = import_module("facet2_inj_ml_model.training")
		return getattr(training_module, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


sys.modules.setdefault("train", sys.modules[__name__])

__version__ = "0.1.0"
__all__ = ["load_model", "BeamOutputModel", "pv_mapping", *_LEGACY_TRAIN_EXPORTS]