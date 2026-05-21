from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from lume.model import LUMEModel
from lume.staged_model import FinalParticlesMixIn
from lume_torch.base import LUMETorchModel
from lume_torch.models.torch_model import TorchModel
import torch
import beamphysics


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Return a NumPy view/copy from tensor-like input on CPU without gradients."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class BeamOutputModel(LUMEModel, FinalParticlesMixIn):
    """
    LUME wrapper around a surrogate model that adds an openPMD beam
    output variable based on a model predicting the beam covariance matrix.

    The surrogate model is expected to support at least the following variables:
    covariance_matrix: TorchNDVariable
        6x6 covariance matrix of the beam distribution in openpmd ParticleBeam order / units.
        Note: The units for openPMD ParticleBeam are meters and eV/c.
    """

    def __init__(
        self, surrogate: TorchModel, n_particles: int = 10000, p0c: float = 1e8, t0: float = 0.0, z0: float = 0.0
    ) -> None:
        super().__init__()
        self.surrogate = LUMETorchModel(surrogate)
        self.n_particles = n_particles
        self.p0c = p0c
        self.t0 = t0
        self.z0 = z0
        self._cache: dict[str, Any] = {"output_beam": None}
        self.set({})  # Initializing with defaults of NN model
        self.update_state()

    def _get(self, names: Iterable[str]) -> dict[str, Any]:
        return {name: self._cache[name] for name in names}

    def _set(self, values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            self._cache[name] = value
        self.surrogate.set(dict(values))
        self.update_state()

    @property
    def supported_variables(self) -> dict[str, Any]:
        return self.surrogate.supported_variables

    def reset(self):
        self.surrogate.reset()
        self._cache = {"output_beam": None}

    def update_state(self):
        self._cache.update(
            self.surrogate.get(list(self.surrogate.supported_variables.keys()))
        )
        covariance_matrix = self._cache["covariance_matrix"]

        particles = torch.distributions.MultivariateNormal(
            loc=torch.zeros(6), covariance_matrix=covariance_matrix
        ).sample((self.n_particles,))

        data = {
            "x": _tensor_to_numpy(particles[:, 0]),
            "y": _tensor_to_numpy(particles[:, 2]),
            "t": _tensor_to_numpy(particles[:, 4] + self.t0),
            "px": _tensor_to_numpy(particles[:, 1]),
            "py": _tensor_to_numpy(particles[:, 3]),
            "pz": _tensor_to_numpy(particles[:, 5] + self.p0c),
            "z": self.z0,
            "weight": _tensor_to_numpy(
                torch.ones(self.n_particles)
            ),
            "status": _tensor_to_numpy(
                torch.ones(self.n_particles, dtype=torch.int32)
            ),
            "species": "electron",
        }
        particle_group = beamphysics.ParticleGroup(data=data)
        self._cache["output_beam"] = particle_group

    @property
    def final_particles(self):
        return self._cache["output_beam"]
