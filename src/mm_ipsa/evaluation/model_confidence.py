"""Model Confidence Set de Hansen, Lunde y Nason (2011).

Los contrastes pareados con correccion de Holm responden si el modelo focal
difiere de cada control por separado. La pregunta de esta investigacion es
distinta: cuales modelos son indistinguibles del mejor. El MCS construye
directamente ese conjunto con cobertura asintotica ``1 - alpha``, de modo que
"ningun modelo domina" pasa de ser una lectura informal de nueve p-valores a un
objeto estadistico con una propiedad declarada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_indices(
    n_observations: int,
    block_size: int,
    samples: int,
    rng: np.random.Generator,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Indices de moving-block bootstrap, opcionalmente confinados por grupo.

    Cuando se entregan ``groups`` cada segmento se remuestrea de forma
    independiente y conserva su largo, por lo que ningun bloque concatena el
    final de un fold con el inicio de otro.
    """
    if block_size < 1:
        raise ValueError("block_size debe ser positivo")
    if samples < 1:
        raise ValueError("samples debe ser positivo")

    if groups is None:
        segments = [np.arange(n_observations)]
    else:
        group_values = np.asarray(list(groups), dtype=object)
        if len(group_values) != n_observations:
            raise ValueError("groups debe tener el largo de las observaciones")
        segments = [
            np.flatnonzero(group_values == group)
            for group in dict.fromkeys(group_values.tolist())
        ]
    if any(len(segment) < block_size for segment in segments):
        raise ValueError("block_size no puede superar el largo de ningun segmento")

    output = np.empty((samples, n_observations), dtype=int)
    for sample in range(samples):
        parts: list[np.ndarray] = []
        for segment in segments:
            length = len(segment)
            picked: list[int] = []
            max_start = length - block_size + 1
            while len(picked) < length:
                start = int(rng.integers(0, max_start))
                picked.extend(range(start, start + block_size))
            parts.append(segment[np.asarray(picked[:length])])
        output[sample] = np.concatenate(parts)
    return output


def _pairwise_statistics(
    losses: np.ndarray,
    bootstrap_index: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estadisticos t pareados observados y su contraparte bootstrap.

    Devuelve la matriz de diferencias medias, los t observados y los t
    remuestreados centrados bajo la hipotesis de igual capacidad predictiva.
    """
    subset = losses[:, active]
    n, m = subset.shape
    mean_loss = subset.mean(axis=0)
    difference = mean_loss[:, None] - mean_loss[None, :]

    # Medias bootstrap por modelo: (samples, m).
    boot_means = subset[bootstrap_index].mean(axis=1)
    boot_difference = boot_means[:, :, None] - boot_means[:, None, :]
    centered = boot_difference - difference[None, :, :]
    variance = (centered**2).mean(axis=0)

    # La diagonal es identicamente cero; se evita 0/0 sin alterar el maximo.
    denominator = np.sqrt(np.maximum(variance, 0.0))
    np.fill_diagonal(denominator, 1.0)
    safe = np.where(denominator > 0.0, denominator, 1.0)

    t_observed = difference / safe
    t_bootstrap = centered / safe[None, :, :]
    np.fill_diagonal(t_observed, 0.0)
    for index in range(m):
        t_bootstrap[:, index, index] = 0.0
    return difference, t_observed, t_bootstrap


def model_confidence_set(
    losses: pd.DataFrame,
    *,
    alpha: float = 0.05,
    block_size: int = 1,
    samples: int = 5_000,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    """Construye el MCS con el estadistico de rango y eliminacion secuencial.

    ``losses`` tiene una fila por observacion y una columna por modelo; todas
    las columnas deben ser perdidas comparables medidas sobre exactamente las
    mismas fechas. El procedimiento repite: contrastar igual capacidad
    predictiva sobre el conjunto vigente y, si se rechaza, eliminar el peor
    modelo segun la regla ``e_max``.

    El p-valor MCS de cada modelo es el maximo acumulado a lo largo de la
    secuencia, garantizando monotonia: un modelo pertenece al conjunto al nivel
    ``alpha`` cuando su p-valor MCS supera ``alpha``.
    """
    if not isinstance(losses, pd.DataFrame):
        raise TypeError("losses debe ser un DataFrame")
    if losses.shape[1] < 2:
        raise ValueError("Se requieren al menos dos modelos")
    if losses.isna().any().any():
        raise ValueError("losses no puede contener valores faltantes")
    values = losses.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("losses contiene valores no finitos")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe pertenecer a (0, 1)")
    if samples < 100:
        raise ValueError("samples debe ser al menos 100")

    names = list(losses.columns)
    n_observations = values.shape[0]
    rng = np.random.default_rng(seed)
    bootstrap_index = block_bootstrap_indices(
        n_observations, block_size, samples, rng, groups
    )

    active = np.arange(len(names))
    records: list[dict[str, object]] = []
    running_pvalue = 0.0
    step = 0

    while len(active) > 1:
        step += 1
        _, t_observed, t_bootstrap = _pairwise_statistics(
            values, bootstrap_index, active
        )
        statistic = float(np.max(np.abs(t_observed)))
        bootstrap_statistic = np.max(np.abs(t_bootstrap), axis=(1, 2))
        pvalue = float(
            (1.0 + np.sum(bootstrap_statistic >= statistic)) / (samples + 1.0)
        )
        running_pvalue = max(running_pvalue, pvalue)

        # Regla e_max: se elimina el modelo cuya peor comparacion es mas
        # desfavorable, es decir el que maximiza sup_j t_ij.
        worst_position = int(np.argmax(t_observed.max(axis=1)))
        eliminated = int(active[worst_position])
        records.append(
            {
                "step": step,
                "model": names[eliminated],
                "mcs_pvalue": running_pvalue,
                "range_statistic": statistic,
                "models_remaining": int(len(active)),
                "eliminated": True,
            }
        )
        active = np.delete(active, worst_position)

    records.append(
        {
            "step": step + 1,
            "model": names[int(active[0])],
            "mcs_pvalue": 1.0,
            "range_statistic": float("nan"),
            "models_remaining": 1,
            "eliminated": False,
        }
    )

    frame = pd.DataFrame(records)
    frame["in_mcs"] = frame["mcs_pvalue"] > alpha
    frame["alpha"] = alpha
    frame["block_size"] = block_size
    frame["bootstrap_samples"] = samples
    mean_losses = losses.mean()
    frame["mean_loss"] = frame["model"].map(mean_losses)
    frame = frame.sort_values("mean_loss").reset_index(drop=True)
    frame["rank_by_mean_loss"] = np.arange(1, len(frame) + 1)
    return frame
