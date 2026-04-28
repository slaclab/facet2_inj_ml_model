"""Compatibility module for loading packaged PyTorch model pickles.

The saved model artifact references ``train.CovarianceSurrogateModel``.
Re-export the inference-relevant symbols from the packaged module so
``torch.load`` can resolve them after installation.
"""

from facet2_model.train import CovarianceAwareLoss
from facet2_model.train import CovarianceSurrogateModel
from facet2_model.train import build_model
from facet2_model.train import chol_vectors_to_covariance

__all__ = [
    "CovarianceAwareLoss",
    "CovarianceSurrogateModel",
    "build_model",
    "chol_vectors_to_covariance",
]