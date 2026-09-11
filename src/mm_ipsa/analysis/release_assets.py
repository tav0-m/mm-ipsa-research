"""Genera activos publicables exclusivamente desde resultados verificados."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

MODEL_LABELS = {
    "MM": "MM-BCD",
    "gaussian_terminal": "Gaussiano",
    "student_t_terminal": "Student-t",
    "historical_weighted": "Histórico EWMA",
    "dcc_garch": "DCC-GARCH",
}
MODEL_COLORS = {
    "MM": "#7C3AED",
    "gaussian_terminal": "#2563EB",
    "student_t_terminal": "#059669",
    "historical_weighted": "#D97706",
    "dcc_garch": "#DC2626",
}


def model_label(name: str) -> str:
    """Etiqueta legible de un modelo, tolerando controles no catalogados."""
    return MODEL_LABELS.get(str(name), str(name))


def model_color(name: str) -> str:
    """Color asignado a un modelo; gris neutro si no esta catalogado."""
    return MODEL_COLORS.get(str(name), "#6B7280")


def package_version() -> str:
    """Version instalada del paquete, para que la portada nunca quede desfasada."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mm-ipsa-research")
    except PackageNotFoundError:
        return "desarrollo"


def count_automated_tests(tests_dir: str | Path = "tests") -> int:
    """Cuenta las pruebas declaradas en el arbol de tests.

    Se calcula en vez de escribirse a mano: una cifra fija en la portada queda
    obsoleta en cuanto se agrega cobertura, y publicarla desactualizada resta
    credibilidad justamente donde se busca demostrarla.
    """
    root = Path(tests_dir)
    if not root.is_dir():
        return 0
    return sum(
        line.strip().startswith("def test_")
        for path in root.glob("test_*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _style_axis(axis: Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    axis.tick_params(colors="#374151")


def plot_crps_stability(scores: pd.DataFrame, output: Path) -> None:
    """Grafica exceso de CRPS frente al mejor modelo dentro de cada fold."""
    data = scores.copy()
    data["crps_excess_bp"] = 10_000 * (
        data["mean_crps"] - data.groupby("fold_id")["mean_crps"].transform("min")
    )
    fold_order = list(dict.fromkeys(data["fold_id"].astype(str).tolist()))
    x = np.arange(len(fold_order))
    figure, axis = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    for model in MODEL_LABELS:
        model_data = (
            data.loc[data["model"] == model]
            .assign(fold_id=lambda frame: frame["fold_id"].astype(str))
            .set_index("fold_id")
            .loc[fold_order]
        )
        values = model_data["crps_excess_bp"].to_numpy()
        axis.plot(
            x,
            values,
            marker="o",
            linewidth=2.4 if model == "MM" else 1.8,
            markersize=7,
            label=model_label(model),
            color=model_color(model),
        )
    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.set_xticks(x, fold_order)
    axis.set_xlabel("Fold de evaluación")
    axis.set_ylabel("Exceso de CRPS frente al mejor modelo del fold (pb)")
    axis.set_title("Estabilidad temporal: menor CRPS es mejor")
    axis.legend(frameon=False, ncol=2)
    _style_axis(axis)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def plot_paired_differences(differences: pd.DataFrame, output: Path) -> None:
    """Grafica MM-control e IC95 en paneles con escalas interpretables."""
    settings = {
        "mean_crps": (10_000, "CRPS: diferencia MM − control (pb)"),
        "energy_score": (10_000, "Energy: diferencia MM − control (×10⁴)"),
        "variogram_score": (1_000, "Variogram: diferencia MM − control (×10³)"),
    }
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    panel_titles = {
        "mean_crps": "CRPS",
        "energy_score": "Energy Score",
        "variogram_score": "Variogram Score",
    }
    for axis, (metric, (scale, title)) in zip(axes, settings.items()):
        subset = differences.loc[differences["metric"] == metric].copy()
        subset["label"] = subset["benchmark_model"].map(MODEL_LABELS)
        subset = subset.iloc[::-1].reset_index(drop=True)
        values = subset["mean_difference"].to_numpy() * scale
        lows = subset["ci_low"].to_numpy() * scale
        highs = subset["ci_high"].to_numpy() * scale
        colors = ["#2563EB" if value < 0 else "#D97706" for value in values]
        axis.errorbar(
            values,
            np.arange(len(subset)),
            xerr=np.vstack([values - lows, highs - values]),
            fmt="none",
            ecolor="#6B7280",
            elinewidth=1.6,
            capsize=4,
        )
        axis.scatter(values, np.arange(len(subset)), c=colors, s=55, zorder=3)
        axis.axvline(0.0, color="#111827", linewidth=1.0)
        axis.set_yticks(np.arange(len(subset)), subset["label"])
        axis.set_xlabel(title)
        axis.set_title(panel_titles[metric])
        axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Validación rolling-origin pareada — negativo favorece a MM-BCD",
        fontsize=14,
    )
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)






















def generate_release_assets(
    rolling_dir: str | Path = "outputs/robustness/rolling_origin",
    destination: str | Path = "research/assets",
) -> list[Path]:
    """Genera las figuras de release a partir de los resultados rolling-origin."""
    source = Path(rolling_dir)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)

    fold_scores = pd.read_csv(source / "probabilistic_scores_by_fold.csv")
    differences = pd.read_csv(source / "probabilistic_score_differences_pooled.csv")
    pooled = pd.read_csv(source / "probabilistic_scores_pooled.csv")

    stability = target / "rolling-origin-crps.png"
    paired = target / "paired-score-differences.png"
    scores = target / "rolling-origin-pooled-scores.csv"

    plot_crps_stability(fold_scores, stability)
    plot_paired_differences(differences, paired)
    pooled.to_csv(scores, index=False)
    return [stability, paired, scores]


if __name__ == "__main__":
    for artifact in generate_release_assets():
        print(artifact)
