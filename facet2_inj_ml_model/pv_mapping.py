"""Affine mapping helpers between machine PV units and simulator parameters."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from botorch.models.transforms.input import AffineInputTransform


PV_MAPPING_BY_SIM_PARAM = {
    "CQ10121:b1_gradient": {
        "experimental_pv": "QUAD:IN10:121:BCTRL",
        "pv_precision": 4,
        "sim_scaling": -2.1,
        "sim_offset": 0.0,
    },
    "GUNF:rf_field_scale": {
        "experimental_pv": "KLYS:LI10:21:AMPL",
        "pv_precision": 6,
        "sim_scaling": 7.89830881e-7,
        "sim_offset": 0.0,
    },
    "GUNF:theta0_deg": {
        "experimental_pv": "KLYS:LI10:21:PHAS",
        "pv_precision": 2,
        "sim_scaling": 1.0,
        "sim_offset": 152.3,
    },
    "SOL10111:solenoid_field_scale": {
        "experimental_pv": "SOLN:IN10:121:BCTRL",
        "pv_precision": 4,
        "sim_scaling": 1.6,
        "sim_offset": 0.0,
    },
    "SQ10122:b1_gradient": {
        "experimental_pv": "QUAD:IN10:122:BCTRL",
        "pv_precision": 4,
        "sim_scaling": -2.1,
        "sim_offset": 0.0,
    },
    "distgen:t_dist:sigma_t:value": {
        "experimental_pv": None,
        "pv_precision": None,
        "sim_scaling": 1.0,
        "sim_offset": 0.0,
    },
    "distgen:total_charge:value": {
        "experimental_pv": "TORO:IN10:591:TMIT_PC",
        "pv_precision": 2,
        "sim_scaling": 1.0,
        "sim_offset": 0.0,
    },
}


def ordered_pv_mapping(feature_cols: Iterable[str]) -> list[dict]:
    ordered_specs = []
    missing = []
    for feature_col in feature_cols:
        spec = PV_MAPPING_BY_SIM_PARAM.get(feature_col)
        if spec is None:
            missing.append(feature_col)
            continue
        ordered_specs.append({"sim_param": feature_col, **spec})
    if missing:
        raise KeyError("Missing PV mapping definitions for: " + ", ".join(missing))
    return ordered_specs


def machine_input_names(feature_cols: Iterable[str]) -> list[str]:
    names = []
    for spec in ordered_pv_mapping(feature_cols):
        names.append(spec["experimental_pv"] or spec["sim_param"])
    return names


def build_pv_to_sim_transform(feature_cols: Iterable[str]) -> AffineInputTransform:
    specs = ordered_pv_mapping(feature_cols)
    coefficient = torch.tensor([spec["sim_scaling"] for spec in specs], dtype=torch.float32)
    offset = torch.tensor([spec["sim_offset"] for spec in specs], dtype=torch.float32)
    return AffineInputTransform(d=len(specs), coefficient=coefficient, offset=offset)


def machine_to_sim_array(machine_values: np.ndarray, feature_cols: Iterable[str]) -> np.ndarray:
    specs = ordered_pv_mapping(feature_cols)
    scales = np.asarray([spec["sim_scaling"] for spec in specs], dtype=np.float32)
    offsets = np.asarray([spec["sim_offset"] for spec in specs], dtype=np.float32)
    return (np.asarray(machine_values, dtype=np.float32) - offsets) / scales


def sim_to_machine_array(sim_values: np.ndarray, feature_cols: Iterable[str]) -> np.ndarray:
    specs = ordered_pv_mapping(feature_cols)
    scales = np.asarray([spec["sim_scaling"] for spec in specs], dtype=np.float32)
    offsets = np.asarray([spec["sim_offset"] for spec in specs], dtype=np.float32)
    return np.asarray(sim_values, dtype=np.float32) * scales + offsets
