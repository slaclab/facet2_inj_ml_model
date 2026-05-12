"""Output transforms for the covariance surrogate model."""

import torch


# M-normalization diagonal used during training
M_DIAG = torch.tensor([1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6], dtype=torch.float32)


class CovarianceDenormTransform(torch.nn.Module):
    """Output transformer that converts M-normalized covariance to physical units.

    Applies C_phys = M_inv @ C_norm @ M_inv^T where M = diag(M_DIAG).
    """

    def __init__(self, m_diag: torch.Tensor = M_DIAG):
        super().__init__()
        self.register_buffer("m_inv_diag", 1.0 / m_diag)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        # cov shape: (..., 6, 6)
        m_inv = torch.diag(self.m_inv_diag)
        return m_inv @ cov @ m_inv.T
