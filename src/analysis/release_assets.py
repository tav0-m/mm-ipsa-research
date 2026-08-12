"""Genera activos publicables exclusivamente desde resultados verificados."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "MM": "MM-BCD",
    "gaussian_terminal": "Gaussiano",
    "student_t_terminal": "Student-t",
    "historical_weighted": "Histórico EWMA",
}
MODEL_COLORS = {
    "MM": "#7C3AED",
    "gaussian_terminal": "#2563EB",
    "student_t_terminal": "#059669",
    "historical_weighted": "#D97706",
}


def _style_axis(axis: plt.Axes) -> None:
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
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
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


def plot_linkedin_card(
    pooled: pd.DataFrame,
    stability: pd.DataFrame,
    metadata: dict,
    output: Path,
) -> None:
    """Genera una portada 1200x627 con conclusiones derivadas de las tablas."""
    best_crps = pooled.sort_values("mean_crps").iloc[0]
    mm_crps = stability.loc[
        (stability["metric"] == "mean_crps") & (stability["model"] == "MM")
    ].iloc[0]
    figure = plt.figure(figsize=(12, 6.27), dpi=100, facecolor="#0B1220")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_axis_off()
    axis.text(
        0.07,
        0.82,
        "Generación de escenarios financieros\ncon Matching Moments",
        color="white",
        fontsize=27,
        fontweight="bold",
        va="top",
    )
    axis.text(
        0.07,
        0.61,
        "Investigación cuantitativa reproducible sobre 15 acciones chilenas",
        color="#CBD5E1",
        fontsize=14,
        va="top",
    )
    facts = [
        (str(metadata["fold_count"]), "folds temporales"),
        (str(metadata["total_evaluation_windows"]), "ventanas OOS H=5"),
        ("45", "tests automatizados"),
    ]
    for index, (value, label) in enumerate(facts):
        x = 0.07 + index * 0.25
        axis.text(x, 0.43, value, color="#A78BFA", fontsize=25, fontweight="bold")
        axis.text(x, 0.35, label, color="#E2E8F0", fontsize=11)
    conclusion = (
        f"Mejor CRPS pooled: {MODEL_LABELS[str(best_crps['model'])]}. "
        f"MM-BCD ganó {int(mm_crps['fold_wins'])}/{int(mm_crps['folds'])} folds en CRPS."
    )
    axis.text(0.07, 0.20, conclusion, color="white", fontsize=13)
    axis.text(
        0.07,
        0.10,
        "Resultado central: ajustar momentos casi exactamente no garantiza superioridad predictiva.",
        color="#FBBF24",
        fontsize=12,
    )
    axis.text(0.93, 0.06, "MM-IPSA Research · v0.4.0", color="#94A3B8", fontsize=10, ha="right")
    figure.savefig(output, dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)


def generate_release_assets(
    rolling_dir: str | Path = "outputs/robustness/rolling_origin",
    destination: str | Path = "docs/assets",
) -> list[Path]:
    import json

    source = Path(rolling_dir)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    fold_scores = pd.read_csv(source / "probabilistic_scores_by_fold.csv")
    differences = pd.read_csv(source / "probabilistic_score_differences_pooled.csv")
    pooled = pd.read_csv(source / "probabilistic_scores_pooled.csv")
    stability = pd.read_csv(source / "model_stability_summary.csv")
    metadata = json.loads((source / "experiment_metadata.json").read_text(encoding="utf-8"))
    outputs = [
        target / "rolling-origin-crps.png",
        target / "paired-score-differences.png",
        target / "linkedin-project-card.png",
    ]
    plot_crps_stability(fold_scores, outputs[0])
    plot_paired_differences(differences, outputs[1])
    plot_linkedin_card(pooled, stability, metadata, outputs[2])
    pooled.to_csv(target / "rolling-origin-pooled-scores.csv", index=False)
    return [*outputs, target / "rolling-origin-pooled-scores.csv"]


if __name__ == "__main__":
    for artifact in generate_release_assets():
        print(artifact)
