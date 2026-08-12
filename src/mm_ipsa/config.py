"""Contratos de configuracion compartidos por pipeline y experimentos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from mm_ipsa.mm.targets import resolve_decay_lambda


def load_config(path: str | Path = "config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    assets = cfg.get("assets", [])
    labels = cfg.get("asset_labels", [])
    if not assets or len(assets) != len(labels) or len(set(labels)) != len(labels):
        raise ValueError("assets y asset_labels deben ser listas no vacias, unicas y del mismo largo")

    data = cfg["data"]
    end_train = pd.Timestamp(data["end_train"])
    start_oos = pd.Timestamp(data["start_oos"])
    if end_train >= start_oos:
        raise ValueError("end_train debe ser estrictamente anterior a start_oos")
    if int(data["H"]) <= 0:
        raise ValueError("H debe ser positivo")

    mm = cfg["mm"]
    decay = resolve_decay_lambda(mm, float(mm.get("observations_per_week", 5.0)))
    if not 0 < decay <= 1:
        raise ValueError("El decaimiento EWMA resuelto debe pertenecer a (0, 1]")
    if int(mm["N_scenarios"]) <= 1 or int(mm["n_starts"]) <= 0:
        raise ValueError("N_scenarios y n_starts no son validos")

    evaluation = cfg["evaluation"]
    if int(evaluation["score_bootstrap_samples"]) < 100:
        raise ValueError("score_bootstrap_samples debe ser al menos 100")
    if int(evaluation["score_bootstrap_block_size"]) <= 0:
        raise ValueError("score_bootstrap_block_size debe ser positivo")
    if not 0.0 < float(evaluation["score_bootstrap_confidence"]) < 1.0:
        raise ValueError("score_bootstrap_confidence debe pertenecer a (0, 1)")
    if evaluation.get("score_multiple_testing") != "holm":
        raise ValueError("score_multiple_testing debe ser holm")
    if int(evaluation["energy_pair_samples"]) <= 0:
        raise ValueError("energy_pair_samples debe ser positivo")


def objective_weights(mm_cfg: dict) -> dict:
    """Construye una unica fuente de verdad para MMObjective."""
    weights = dict(mm_cfg["moment_weights"])
    for key in (
        "cov_weight",
        "cov_normalize",
        "cov_scale_floor",
        "cov_offdiag_only",
        "moment_scale_mode",
        "moment_scale_floor",
    ):
        if key in mm_cfg:
            weights[key] = mm_cfg[key]
    return weights


def target_parameters(mm_cfg: dict) -> dict:
    observations = float(mm_cfg.get("observations_per_week", 5.0))
    return {
        "decay_lambda": resolve_decay_lambda(mm_cfg, observations),
        "covariance_shrinkage": float(mm_cfg.get("covariance_shrinkage", 0.0)),
    }
