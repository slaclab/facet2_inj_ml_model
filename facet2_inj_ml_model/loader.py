"""Model loading utilities"""

import os
from pathlib import Path
from lume_torch.models import TorchModel


_MODEL_CONFIGS = {
    "machine": "lumetorchyaml-machine/injector_machine.yaml",
    "sim": "lumetorchyaml-sim/injector_simulator.yaml",
}


def get_resource_path(filename):
    """Get the absolute path to a resource file."""
    package_dir = Path(__file__).parent
    resource_path = package_dir / "resources" / filename
    
    if not resource_path.exists():
        raise FileNotFoundError(f"Resource file not found: {resource_path}")
    
    return str(resource_path)


def load_model(input_space="machine"):
    """
    Load the FACET TorchModel.

    Parameters
    ----------
    input_space : str, optional
        Which input space the model expects. ``"machine"`` (default) accepts
        machine PV values; ``"sim"`` accepts simulator parameters.

    Returns:
        TorchModel: Loaded model instance ready for inference.
    
    Example:
        >>> from facet2_inj_ml_model import load_model
        >>> model = load_model()                    # machine-PV inputs
        >>> model_sim = load_model("sim")           # simulator-parameter inputs

    """
    if input_space not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown input_space {input_space!r}; expected one of {list(_MODEL_CONFIGS)}"
        )
    config_path = get_resource_path(_MODEL_CONFIGS[input_space])
    
    # Load the model using the config
    model = TorchModel(config_path)
    
    return model