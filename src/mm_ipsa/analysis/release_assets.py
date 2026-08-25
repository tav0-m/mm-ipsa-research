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
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
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
        (str(count_automated_tests()), "tests automatizados"),
    ]
    for index, (value, label) in enumerate(facts):
        x = 0.07 + index * 0.25
        axis.text(x, 0.43, value, color="#A78BFA", fontsize=25, fontweight="bold")
        axis.text(x, 0.35, label, color="#E2E8F0", fontsize=11)
    conclusion = (
        f"Mejor CRPS pooled: {model_label(best_crps['model'])}. "
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
    axis.text(
        0.93,
        0.06,
        f"MM-IPSA Research · v{package_version()}",
        color="#94A3B8",
        fontsize=10,
        ha="right",
    )
    figure.savefig(output, dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_pooled_crps(pooled: pd.DataFrame, output: Path) -> None:
    """Compara el CRPS pooled como exceso respecto del mejor modelo."""
    data = pooled.sort_values("mean_crps", ascending=False).copy()
    best = float(data["mean_crps"].min())
    winner = model_label(str(data.loc[data["mean_crps"].idxmin(), "model"]))
    data["excess_bp"] = 10_000 * (data["mean_crps"] - best)
    labels = [model_label(model) for model in data["model"]]
    colors = [model_color(model) for model in data["model"]]

    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    bars = axis.barh(labels, data["excess_bp"], color=colors, alpha=0.9)
    maximum = max(float(data["excess_bp"].max()), 0.1)
    for bar, excess, score in zip(bars, data["excess_bp"], data["mean_crps"]):
        axis.text(
            float(excess) + maximum * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"Δ {float(excess):.2f} pb  |  CRPS {float(score):.5f}",
            va="center",
            fontsize=11,
            color="#111827",
        )
    axis.set_xlim(0, maximum * 1.55)
    axis.set_xlabel("Exceso de CRPS respecto del mejor modelo (puntos base)")
    axis.set_title(
        f"Resultado pooled fuera de muestra\n{winner} obtuvo el menor CRPS",
        fontsize=17,
        fontweight="bold",
        pad=20,
    )
    axis.text(
        0.0,
        1.01,
        "169 ventanas OOS de cinco días · menor es mejor",
        transform=axis.transAxes,
        fontsize=11,
        color="#4B5563",
    )
    _style_axis(axis)
    figure.savefig(output, dpi=120, facecolor="white")
    plt.close(figure)


def plot_crps_stability_square(scores: pd.DataFrame, output: Path) -> None:
    """Muestra la estabilidad del exceso de CRPS en los cuatro folds."""
    data = scores.copy()
    data["crps_excess_bp"] = 10_000 * (
        data["mean_crps"] - data.groupby("fold_id")["mean_crps"].transform("min")
    )
    fold_order = list(dict.fromkeys(data["fold_id"].astype(str).tolist()))
    x = np.arange(len(fold_order))
    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    for model in MODEL_LABELS:
        model_data = (
            data.loc[data["model"] == model]
            .assign(fold_id=lambda frame: frame["fold_id"].astype(str))
            .set_index("fold_id")
            .loc[fold_order]
        )
        axis.plot(
            x,
            model_data["crps_excess_bp"].to_numpy(dtype=float),
            marker="o",
            linewidth=3.0 if model == "MM" else 2.0,
            markersize=9,
            label=model_label(model),
            color=model_color(model),
        )
    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.set_xticks(x, fold_order)
    axis.set_xlabel("Fold de evaluación")
    axis.set_ylabel("Exceso de CRPS frente al ganador del fold (pb)")
    axis.set_title(
        "Estabilidad temporal del desempeño predictivo\nMM-BCD ganó 0 de 4 folds en CRPS",
        fontsize=17,
        fontweight="bold",
        pad=20,
    )
    axis.legend(frameon=False, ncol=2, loc="upper left")
    _style_axis(axis)
    figure.savefig(output, dpi=120, facecolor="white")
    plt.close(figure)


def plot_paired_differences_square(differences: pd.DataFrame, output: Path) -> None:
    """Forest plot vertical de diferencias pareadas MM menos benchmark."""
    settings = {
        "mean_crps": (10_000, "CRPS (pb)"),
        "energy_score": (10_000, "Energy Score (×10⁴)"),
        "variogram_score": (1_000, "Variogram Score (×10³)"),
    }
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    for axis, (metric, (scale, title)) in zip(axes, settings.items()):
        subset = differences.loc[differences["metric"] == metric].copy().iloc[::-1]
        labels = subset["benchmark_model"].map(MODEL_LABELS).tolist()
        values = subset["mean_difference"].to_numpy(dtype=float) * scale
        lows = subset["ci_low"].to_numpy(dtype=float) * scale
        highs = subset["ci_high"].to_numpy(dtype=float) * scale
        y = np.arange(len(subset))
        axis.errorbar(
            values,
            y,
            xerr=np.vstack([values - lows, highs - values]),
            fmt="none",
            ecolor="#6B7280",
            elinewidth=1.8,
            capsize=5,
        )
        colors = ["#7C3AED" if value < 0 else "#D97706" for value in values]
        axis.scatter(values, y, c=colors, s=75, zorder=3)
        axis.axvline(0.0, color="#111827", linewidth=1.0)
        axis.set_yticks(y, labels)
        axis.set_xlabel(f"Diferencia MM − benchmark: {title}")
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Inferencia pareada rolling-origin\nNegativo favorece a MM-BCD · barras = IC 95%",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(output, dpi=120, facecolor="white")
    plt.close(figure)


def plot_calibration_vs_prediction(
    fold_scores: pd.DataFrame,
    calibration: pd.DataFrame,
    output: Path,
) -> None:
    """Contrasta ajuste in-sample y exceso predictivo OOS sin imponer tendencia."""
    mm = fold_scores.loc[fold_scores["model"] == "MM", ["fold_id", "mean_crps"]].copy()
    winners = fold_scores.groupby("fold_id")["mean_crps"].min().rename("best_crps")
    mm = mm.join(winners, on="fold_id")
    mm["crps_excess_bp"] = 10_000 * (mm["mean_crps"] - mm["best_crps"])
    data = mm.merge(calibration[["fold_id", "F"]], on="fold_id", validate="one_to_one")
    x = data["F"].to_numpy(dtype=float) * 1e10
    y = data["crps_excess_bp"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    axis.scatter(x, y, s=150, color=MODEL_COLORS["MM"], zorder=3)
    for x_value, y_value, fold in zip(x, y, data["fold_id"].astype(str)):
        axis.annotate(
            fold,
            (x_value, y_value),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=11,
        )
    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.set_xlabel("Objetivo de calibración F (×10⁻¹⁰; menor es mejor)")
    axis.set_ylabel("Exceso de CRPS OOS frente al mejor modelo del fold (pb)")
    axis.set_title(
        "Calibrar casi exactamente no garantizó ganar OOS\nLos cuatro folds convergieron y fueron estacionarios",
        fontsize=17,
        fontweight="bold",
        pad=20,
    )
    _style_axis(axis)
    figure.savefig(output, dpi=120, facecolor="white")
    plt.close(figure)


def _linkedin_figure(subtitle: str, title: str):
    """Lienzo cuadrado para una evidencia cuantitativa del carrusel."""
    figure = plt.figure(figsize=(10, 10), dpi=120, facecolor="white")
    figure.text(0.06, 0.95, subtitle, fontsize=12, color="#6B7280", va="top")
    figure.text(
        0.06,
        0.90,
        title,
        fontsize=24,
        fontweight="bold",
        color="#111827",
        va="top",
    )
    return figure


def plot_linkedin_scorecard(pooled: pd.DataFrame, output: Path) -> None:
    """Tabla compacta de las tres reglas propias bajo el protocolo principal."""
    metrics = ["mean_crps", "energy_score", "variogram_score"]
    data = pooled.sort_values("mean_crps").reset_index(drop=True)
    rows = [
        [
            model_label(str(row["model"])),
            f"{float(row['mean_crps']):.5f}",
            f"{float(row['energy_score']):.5f}",
            f"{float(row['variogram_score']):.4f}",
        ]
        for _, row in data.iterrows()
    ]
    figure = _linkedin_figure(
        "Rolling-origin · 169 ventanas OOS · horizonte 5 días · menor es mejor",
        "DCC-GARCH lidera las 3 métricas",
    )
    axis = figure.add_axes((0.055, 0.12, 0.89, 0.66))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Modelo", "CRPS", "Energy", "Variogram"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.34, 0.22, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 3.1)
    for column in range(4):
        cell = table[(0, column)]
        cell.set_facecolor("#111827")
        cell.set_edgecolor("white")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    for row_index, (_, row) in enumerate(data.iterrows(), start=1):
        name = str(row["model"])
        for column in range(4):
            cell = table[(row_index, column)]
            cell.set_edgecolor("#E5E7EB")
            cell.set_linewidth(0.8)
            cell.set_facecolor("#F9FAFB" if row_index % 2 else "white")
            cell.get_text().set_color("#111827")
        table[(row_index, 0)].get_text().set_color(model_color(name))
        table[(row_index, 0)].get_text().set_fontweight("bold")
        for metric_column, metric in enumerate(metrics, start=1):
            if np.isclose(float(row[metric]), float(data[metric].min())):
                best_cell = table[(row_index, metric_column)]
                best_cell.set_facecolor("#DCFCE7")
                best_cell.get_text().set_fontweight("bold")
    figure.text(
        0.06,
        0.08,
        "Los tres scores evalúan dimensiones complementarias de la distribución.",
        fontsize=11.5,
        color="#6B7280",
    )
    figure.savefig(output, dpi=120, facecolor="white")
    plt.close(figure)


def plot_linkedin_temporal_ranks(scores: pd.DataFrame, output: Path) -> None:
    """Matriz de rangos CRPS que hace visible la excepción temporal de 2024."""
    fold_order = list(dict.fromkeys(scores["fold_id"].astype(str).tolist()))
    weighted = scores.assign(
        weighted_loss=scores["mean_crps"] * scores["n_observations"]
    )
    pooled_order = (
        weighted.groupby("model")[["weighted_loss", "n_observations"]]
        .sum()
        .assign(score=lambda frame: frame["weighted_loss"] / frame["n_observations"])
        .sort_values("score")
        .index.tolist()
    )
    matrix = (
        scores.assign(fold_id=scores["fold_id"].astype(str))
        .pivot(index="model", columns="fold_id", values="mean_crps")
        .loc[pooled_order, fold_order]
        .rank(axis=0, method="min")
    )
    rank_values = matrix.to_numpy(dtype=float)
    dcc_row = pooled_order.index("dcc_garch")
    dcc_wins = int(np.sum(rank_values[dcc_row] == 1))

    figure = _linkedin_figure(
        "Rango por CRPS dentro de cada fold · 1 = mejor",
        f"DCC-GARCH gana {dcc_wins} de {len(fold_order)} períodos",
    )
    axis = figure.add_axes((0.24, 0.18, 0.68, 0.58))
    image = axis.imshow(rank_values, cmap="RdYlGn_r", vmin=1, vmax=5, aspect="auto")
    del image
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            rank = int(rank_values[row, column])
            axis.text(
                column,
                row,
                f"{rank}.º",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold" if rank == 1 else "normal",
                color="#111827",
            )
    axis.set_xticks(np.arange(len(fold_order)), [fold.replace("H1", " H1") for fold in fold_order])
    axis.set_yticks(np.arange(len(pooled_order)), [model_label(name) for name in pooled_order])
    axis.tick_params(axis="x", labelsize=13, length=0, pad=10)
    axis.tick_params(axis="y", labelsize=13, length=0, pad=10)
    axis.set_xticks(np.arange(-0.5, len(fold_order), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(pooled_order), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=3)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.text(
        0.06,
        0.08,
        "Student-t gana 2024; el liderazgo de DCC-GARCH no es uniforme.",
        fontsize=11.5,
        color="#6B7280",
    )
    figure.savefig(output, dpi=120, facecolor="white")
    plt.close(figure)


def plot_linkedin_paired_mm_vs_dcc(differences: pd.DataFrame, output: Path) -> None:
    """Forest plot comparable entre métricas mediante diferencias relativas."""
    order = ["mean_crps", "energy_score", "variogram_score"]
    labels = {"mean_crps": "CRPS", "energy_score": "Energy", "variogram_score": "Variogram"}
    data = (
        differences.loc[differences["benchmark_model"] == "dcc_garch"]
        .set_index("metric")
        .loc[order]
        .reset_index()
    )
    denominator = data["benchmark_mean_loss"].to_numpy(dtype=float)
    values = 100 * data["mean_difference"].to_numpy(dtype=float) / denominator
    lows = 100 * data["ci_low"].to_numpy(dtype=float) / denominator
    highs = 100 * data["ci_high"].to_numpy(dtype=float) / denominator
    positions = np.arange(len(data))

    figure = _linkedin_figure(
        "Diferencia relativa MM-BCD − DCC-GARCH · IC95%",
        "MM-BCD queda atrás en las 3 métricas",
    )
    axis = figure.add_axes((0.20, 0.18, 0.70, 0.58))
    axis.errorbar(
        values,
        positions,
        xerr=np.vstack([values - lows, highs - values]),
        fmt="none",
        ecolor="#6B7280",
        elinewidth=2.2,
        capsize=7,
    )
    axis.scatter(values, positions, s=160, color=model_color("MM"), zorder=3)
    axis.axvline(0.0, color="#111827", linewidth=1.4)
    span = max(float(highs.max()), 1.0)
    for position, value, high, pvalue in zip(
        positions,
        values,
        highs,
        data["pvalue_holm"].to_numpy(dtype=float),
    ):
        axis.text(
            float(high) + span * 0.035,
            position,
            f"+{float(value):.2f}% · p={float(pvalue):.3f}",
            va="center",
            fontsize=12.5,
            color="#374151",
        )
    axis.set_yticks(positions, [labels[metric] for metric in order])
    axis.invert_yaxis()
    axis.set_xlim(min(-0.3, float(lows.min()) - 0.2), span * 1.42)
    axis.set_xlabel("Pérdida relativa de MM-BCD frente a DCC-GARCH (%)", fontsize=12)
    axis.tick_params(labelsize=13, colors="#111827", length=0)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.text(
        0.06,
        0.08,
        "Los tres intervalos quedan sobre cero tras corrección de Holm.",
        fontsize=11.5,
        color="#6B7280",
    )
    figure.savefig(output, dpi=120, facecolor="white")
    plt.close(figure)


def plot_linkedin_calibration_map(
    calibration_by_fold: pd.DataFrame,
    fold_scores: pd.DataFrame,
    output: Path,
) -> None:
    """Resume calibración rolling-origin ponderando cada fold por sus ventanas."""
    weights = fold_scores[["fold_id", "model", "n_observations"]]
    merged = calibration_by_fold.merge(weights, on=["fold_id", "model"], validate="one_to_one")
    rows: list[dict[str, float | str]] = []
    for name, group in merged.groupby("model"):
        weight = group["n_observations"].to_numpy(dtype=float)
        rows.append(
            {
                "model": str(name),
                "scale_error_pct": float(
                    100 * abs(np.average(group["mean_dispersion_ratio"], weights=weight) - 1)
                ),
                "reliability": float(
                    np.average(group["mean_reliability_index"], weights=weight)
                ),
            }
        )
    data = pd.DataFrame(rows)

    figure = _linkedin_figure(
        "Promedio rolling-origin ponderado · menor en ambos ejes es mejor",
        "Ajustar la escala no ajusta la distribución",
    )
    axis = figure.add_axes((0.16, 0.18, 0.74, 0.58))
    offsets = {
        "MM": (10, -20),
        "historical_weighted": (10, 10),
        "student_t_terminal": (-90, -20),
        "gaussian_terminal": (10, 10),
        "dcc_garch": (-105, 10),
    }
    for _, row in data.iterrows():
        name = str(row["model"])
        x_value = float(row["scale_error_pct"])
        y_value = float(row["reliability"])
        axis.scatter(
            x_value,
            y_value,
            s=190,
            color=model_color(name),
            edgecolors="white",
            linewidths=2,
            zorder=3,
        )
        axis.annotate(
            model_label(name),
            (x_value, y_value),
            xytext=offsets[name],
            textcoords="offset points",
            fontsize=12,
            color="#111827",
        )
    axis.set_xlabel("Error de dispersión frente al ideal (%)", fontsize=12)
    axis.set_ylabel("Índice de fiabilidad PIT", fontsize=12)
    axis.tick_params(labelsize=11.5, colors="#374151")
    axis.grid(color="#E5E7EB", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.06,
        0.08,
        "MM-BCD acierta la escala; DCC-GARCH describe mejor la forma del PIT.",
        fontsize=11.5,
        color="#6B7280",
    )
    figure.savefig(output, dpi=120, facecolor="white")
    plt.close(figure)


def generate_release_assets(
    rolling_dir: str | Path = "outputs/robustness/rolling_origin",
    destination: str | Path = "docs/assets",
) -> list[Path]:
    """Genera las figuras de release a partir de los resultados rolling-origin."""
    import json

    source = Path(rolling_dir)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    fold_scores = pd.read_csv(source / "probabilistic_scores_by_fold.csv")
    differences = pd.read_csv(source / "probabilistic_score_differences_pooled.csv")
    pooled = pd.read_csv(source / "probabilistic_scores_pooled.csv")
    stability = pd.read_csv(source / "model_stability_summary.csv")
    calibration = pd.read_csv(source / "fold_calibration.csv")
    calibration_pit = pd.read_csv(source / "calibration_pit_by_fold.csv")
    metadata = json.loads((source / "experiment_metadata.json").read_text(encoding="utf-8"))
    outputs = [
        target / "rolling-origin-crps.png",
        target / "paired-score-differences.png",
        target / "linkedin-project-card.png",
        target / "linkedin-01-cover.png",
        target / "linkedin-02-pooled-crps.png",
        target / "linkedin-03-temporal-crps.png",
        target / "linkedin-04-paired-inference.png",
        target / "linkedin-05-calibration-vs-prediction.png",
    ]
    plot_crps_stability(fold_scores, outputs[0])
    plot_paired_differences(differences, outputs[1])
    plot_linkedin_card(pooled, stability, metadata, outputs[2])
    plot_linkedin_card(pooled, stability, metadata, outputs[3])
    plot_pooled_crps(pooled, outputs[4])
    plot_crps_stability_square(fold_scores, outputs[5])
    plot_paired_differences_square(differences, outputs[6])
    plot_calibration_vs_prediction(fold_scores, calibration, outputs[7])

    linkedin_dir = target / "linkedin-v07"
    linkedin_dir.mkdir(parents=True, exist_ok=True)
    linkedin_outputs = [
        linkedin_dir / "01-comparacion-modelos.png",
        linkedin_dir / "02-estabilidad-temporal.png",
        linkedin_dir / "03-inferencia-mm-vs-dcc.png",
        linkedin_dir / "04-calibracion-distributiva.png",
    ]
    plot_linkedin_scorecard(pooled, linkedin_outputs[0])
    plot_linkedin_temporal_ranks(fold_scores, linkedin_outputs[1])
    plot_linkedin_paired_mm_vs_dcc(differences, linkedin_outputs[2])
    plot_linkedin_calibration_map(calibration_pit, fold_scores, linkedin_outputs[3])

    pooled.to_csv(target / "rolling-origin-pooled-scores.csv", index=False)
    return [*outputs, *linkedin_outputs, target / "rolling-origin-pooled-scores.csv"]


if __name__ == "__main__":
    for artifact in generate_release_assets():
        print(artifact)
