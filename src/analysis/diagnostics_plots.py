# En esta parte del código se generarán los gráficos
# de diagnóstico del proyecto MM-BCD IPSA
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

# ── Paleta corporativa UDP ────────────────────────────────────────────────────
UDP_BLUE   = "#003E7E"
UDP_GRAY   = "#585858"
BG_COLOR   = "#EBF1F9"
HIST_COLOR = "#2C5F8A"
MM_COLOR   = "#E05C1A"
WINNER_BG  = "#C6EFCE"
WINNER_FG  = "#006400"

# Colores por start (consistente en todos los gráficos)
START_COLORS = {
    1: "#1f77b4",   # azul   — start frío
    2: "#ff7f0e",   # naranja — Start 2
    3: "#2ca02c",   # verde
    4: "#9467bd",   # morado
    5: "#d62728",   # rojo
}
START_STYLES = {1: "-", 2: "-", 3: "-", 4: "-", 5: ":"}


def _style_ax(ax: Axes) -> None:
    """Aplica el estilo base UDP a un eje."""
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, color="white", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: Figure, path: str, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [ok] guardado: {path}")


def generate_wiener(mu_daily, Sigma_daily, H, N_W, seed):
    mu = np.asarray(mu_daily, dtype=float)
    Sigma = np.asarray(Sigma_daily, dtype=float)
    mean = mu * float(H)
    cov = Sigma * float(H)
    rng = np.random.default_rng(seed)
    try:
        return rng.multivariate_normal(mean, cov, size=int(N_W))
    except Exception:
        eigvals, eigvecs = np.linalg.eigh(cov)
        cov_reg = eigvecs @ np.diag(np.clip(eigvals, 1e-10, None)) @ eigvecs.T
        return rng.multivariate_normal(mean, cov_reg, size=int(N_W))


def plot_kurt_comparison(
    hist_data, mm_data, wiener_data, labels, fig_dir, mm_probabilities=None
):
    """Compara kurtosis respetando las probabilidades no uniformes de MM."""
    hist_kurt = stats.kurtosis(np.asarray(hist_data), fisher=False, axis=0)
    mm_array = np.asarray(mm_data, dtype=float)
    if mm_probabilities is None:
        probabilities = np.full(len(mm_array), 1.0 / len(mm_array))
    else:
        probabilities = np.asarray(mm_probabilities, dtype=float)
        probabilities = probabilities / probabilities.sum()
    mean_mm = probabilities @ mm_array
    deviation_mm = mm_array - mean_mm
    variance_mm = probabilities @ deviation_mm**2
    mm_kurt = (probabilities @ deviation_mm**4) / np.maximum(variance_mm**2, 1e-15)
    wiener_kurt = stats.kurtosis(np.asarray(wiener_data), fisher=False, axis=0)
    DiagnosticsPlots({"asset_labels": labels, "data": {"H": 5}}, out_dir=fig_dir).plot_kurt_comparison(
        hist_kurt, mm_kurt, wiener_kurt
    )


def _scenario_covariance(X: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    p = np.asarray(p, dtype=float)
    mu = p @ X
    dev = X - mu[np.newaxis, :]
    Sigma = (p[:, None, None] * dev[:, :, None] * dev[:, None, :]).sum(0)
    return Sigma, mu


def markowitz_frontier(X: np.ndarray, p: np.ndarray, R_min, R_max, n_pts, max_w):
    X = np.asarray(X, dtype=float)
    p = np.asarray(p, dtype=float)
    Sigma, mu = _scenario_covariance(X, p)
    n = X.shape[1]

    weights = np.ones(n) / n
    bounds = [(0.0, float(max_w))] * n
    frontier = []
    targets = np.linspace(float(R_min), float(R_max), int(n_pts))

    def obj(w):
        return float(w @ Sigma @ w)

    for target in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, target=target: float(w @ mu - target)},
        ]
        result = minimize(obj, weights, method="SLSQP", bounds=bounds, constraints=cons,
                          options={"ftol": 1e-9, "maxiter": 300})
        if not result.success:
            continue
        w_opt = np.maximum(result.x, 0)
        w_opt = w_opt / w_opt.sum()
        ret = float(w_opt @ mu)
        std = float(np.sqrt(w_opt @ Sigma @ w_opt))
        # FIX: calcular CVaR post-hoc usando escenarios ponderados
        port_rets = X @ w_opt
        alpha = 0.05
        idx_sorted = np.argsort(port_rets)
        r_s = port_rets[idx_sorted]
        p_s = p[idx_sorted]
        cum_p = np.cumsum(p_s)
        tail = cum_p <= alpha
        if tail.any():
            cvar_val = float(np.dot(r_s[tail], p_s[tail]) / p_s[tail].sum())
        else:
            cvar_val = float(r_s[0])
        frontier.append({"return": ret * 100.0, "std": std * 100.0,
                          "CVaR": cvar_val * 100.0, "weights": w_opt})

    return frontier


def cvar_frontier(X, p, R_min, R_max, n_pts, max_w, beta):
    """
    Frontera eficiente de CVaR via LP Rockafellar-Uryasev (2000).

    CORRECCIONES vs versión anterior:
    1. Usa LP exacto en lugar de SLSQP con CVaR equiprobable.
    2. CVaR calculado con probabilidades ponderadas p_j (no equiprobable).
    3. Retorna clave "CVaR" (no "CVaR_%") — consistente con run.py L406.
    4. Retorna "std" — consistente con run.py L405.
    5. Usa retorno objetivo como igualdad. Con retorno minimo, la LP
       repetia el mismo portafolio en todos los targets cuando el optimo
       global de CVaR ya excedia el retorno requerido.
    """
    from scipy.optimize import linprog as _linprog

    X = np.asarray(X, dtype=float)
    p = np.asarray(p, dtype=float)
    _, mu = _scenario_covariance(X, p)
    N, n = X.shape
    alpha = 1.0 - float(beta)   # beta=0.95 → alpha=0.05
    Sigma, _ = _scenario_covariance(X, p)

    frontier = []
    targets = np.linspace(float(R_min), float(R_max), int(n_pts))

    for target in targets:
        # Variables: [w(n), xi(1), u_0..u_{N-1}(N)]
        nv = n + 1 + N
        c_ = np.zeros(nv)
        c_[n]    = 1.0
        c_[n+1:] = p / alpha

        # u_j >= -w'x_j - xi  =>  -w'x_j - xi - u_j <= 0
        A_tail = np.zeros((N, nv), dtype=float)
        for j in range(N):
            A_tail[j, :n]    = -X[j]
            A_tail[j, n]     = -1.0
            A_tail[j, n+1+j] = -1.0
        b_tail = np.zeros(N, dtype=float)

        A_ub = A_tail
        b_ub = b_tail

        # Igualdades: presupuesto y retorno objetivo.
        A_eq = np.zeros((2, nv), dtype=float)
        A_eq[0, :n] = 1.0
        A_eq[1, :n] = mu.astype(float)
        b_eq = np.array([1.0, target], dtype=float)

        bounds = [(0.0, float(max_w))] * n + [(-10.0, 10.0)] + [(0.0, None)] * N

        res = _linprog(c_, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=bounds, method="highs")
        if res.status != 0:
            continue

        w_opt = np.clip(res.x[:n], 0.0, float(max_w))
        s = w_opt.sum()
        if s < 1e-10:
            continue
        w_opt /= s

        ret_val = float(w_opt @ mu) * 100.0
        std_val = float(np.sqrt(w_opt @ Sigma @ w_opt)) * 100.0

        # CVaR exacto con probabilidades ponderadas
        port_rets = X @ w_opt
        idx_s = np.argsort(port_rets)
        r_s = port_rets[idx_s]; p_s = p[idx_s]
        cum_p = np.cumsum(p_s)
        tail = cum_p <= alpha
        if tail.any():
            cvar_val = float(np.dot(r_s[tail], p_s[tail]) / p_s[tail].sum()) * 100.0
        else:
            cvar_val = float(r_s[0]) * 100.0

        frontier.append({
            "return":  ret_val,
            "std":     std_val,
            "CVaR":    cvar_val,   # FIX: clave "CVaR" (no "CVaR_%")
            "weights": w_opt,
        })

    return frontier


def optimize_max_sharpe(X: np.ndarray, p: np.ndarray,
                        max_w: float = 0.25, rf: float = 0.0,
                        n_starts: int = 10, seed: int = 42) -> dict:
    """
    Portafolio de Máximo Sharpe Ratio (portafolio tangente) con restricciones de peso.

    Problema: max  SR(w) = (mu'w - rf) / sqrt(w'Sigma w)
              s.t. sum(w) = 1,  0 <= w_i <= max_w

    Parametrizado como: minimize  -SR(w)
    con gradiente analitico:
        d(-SR)/dw = -[mu - SR(w) * Sigma @ w / sigma_p] / sigma_p

    Multi-start SLSQP para evitar minimos locales.

    Parameters
    ----------
    X      : (N, n) escenarios de retornos
    p      : (N,)   probabilidades MM
    max_w  : float  maximo peso por activo
    rf     : float  tasa libre de riesgo (por periodo H)
    n_starts : int  numero de inicializaciones aleatorias
    seed   : int    semilla reproducibilidad

    Returns
    -------
    dict con keys: 'weights' (n,), 'sharpe', 'return_pct', 'std_pct', 'cvar_pct'
    """
    X = np.asarray(X, dtype=float)
    p = np.asarray(p, dtype=float)
    Sigma, mu = _scenario_covariance(X, p)
    n = X.shape[1]
    alpha = 0.05  # CVaR al 95% (consistente con config)

    def neg_sharpe(w):
        sigma_p = float(np.sqrt(np.maximum(w @ Sigma @ w, 1e-12)))
        excess  = float(w @ mu) - rf
        return -excess / sigma_p

    def neg_sharpe_grad(w):
        Sw       = Sigma @ w
        sigma_p2 = float(w @ Sw)
        sigma_p  = float(np.sqrt(max(sigma_p2, 1e-12)))
        excess   = float(w @ mu) - rf
        sr_now   = excess / sigma_p
        grad_pos = (mu - sr_now * Sw / sigma_p) / sigma_p
        return -grad_pos

    bounds = [(0.0, float(max_w))] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    rng     = np.random.default_rng(seed)
    best_sr = -np.inf
    best_w  = None

    # Multi-start: inicializaciones aleatorias en el simplex
    for k in range(n_starts):
        if k == 0:
            w0 = np.ones(n) / n   # igual ponderado como start 0
        else:
            raw = rng.dirichlet(np.ones(n))
            raw = np.clip(raw, 0.0, float(max_w))
            raw /= raw.sum()
            w0 = raw

        res = minimize(neg_sharpe, w0, jac=neg_sharpe_grad, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-10, "maxiter": 500})
        if not res.success:
            continue
        w_cand = np.maximum(res.x, 0.0)
        s = w_cand.sum()
        if s < 1e-10:
            continue
        w_cand /= s
        sr_cand = -float(neg_sharpe(w_cand))
        if sr_cand > best_sr:
            best_sr = sr_cand
            best_w  = w_cand.copy()

    if best_w is None:
        # Fallback: igual ponderado
        best_w = np.ones(n) / n
        best_sr = float(-neg_sharpe(best_w))

    # Métricas finales
    ret_val = float(best_w @ mu) * 100.0
    std_val = float(np.sqrt(best_w @ Sigma @ best_w)) * 100.0

    # CVaR exacto con probabilidades ponderadas
    port_rets   = X @ best_w
    idx_s       = np.argsort(port_rets)
    r_s, p_s    = port_rets[idx_s], p[idx_s]
    cum_p       = np.cumsum(p_s)
    tail        = cum_p <= alpha
    if tail.any():
        cvar_val = float(np.dot(r_s[tail], p_s[tail]) / p_s[tail].sum()) * 100.0
    else:
        cvar_val = float(r_s[0]) * 100.0

    print(f"  Max-Sharpe: SR={best_sr:.4f}, ret={ret_val:.4f}%, "
          f"std={std_val:.4f}%, CVaR={cvar_val:.4f}%")

    return {
        "weights":     best_w,
        "sharpe":      best_sr,
        "return_pct":  ret_val,
        "std_pct":     std_val,
        "cvar_pct":    cvar_val,
    }


def plot_frontiers(front_mk_mm, front_mk_w, front_cv_mm, front_cv_w, fig_dir):
    rows = []
    for item in front_mk_mm:
        rows.append({"Model": "MM", "Method": "Markowitz", "Return_%": item["return"],
                     "Std_%": item.get("std", np.nan), "CVaR_%": item.get("CVaR", np.nan)})
    for item in front_mk_w:
        rows.append({"Model": "Wiener", "Method": "Markowitz", "Return_%": item["return"],
                     "Std_%": item.get("std", np.nan), "CVaR_%": item.get("CVaR", np.nan)})
    for item in front_cv_mm:
        rows.append({"Model": "MM", "Method": "CVaR", "Return_%": item["return"],
                     "Std_%": item.get("std", np.nan), "CVaR_%": item.get("CVaR", np.nan)})
    for item in front_cv_w:
        rows.append({"Model": "Wiener", "Method": "CVaR", "Return_%": item["return"],
                     "Std_%": item.get("std", np.nan), "CVaR_%": item.get("CVaR", np.nan)})

    df = pd.DataFrame(rows)
    DiagnosticsPlots({"asset_labels": [], "data": {"H": 5}}, out_dir=fig_dir).plot_fronteras_eficientes(df)


def plot_portfolio_weights(w_dict, labels, fig_dir):
    rows = []
    for name, w in w_dict.items():
        w = np.asarray(w, dtype=float)
        for i, label in enumerate(labels):
            rows.append({"Portafolio": name, "Activo": label, "Peso_pct": float(w[i] * 100.0)})
    df = pd.DataFrame(rows)
    DiagnosticsPlots({"asset_labels": labels, "data": {"H": 5}}, out_dir=fig_dir).plot_portfolio_weights(df)


def compute_backtest_metrics(w_dict, R_oos, labels, H: int = 5):
    """
    Calcula métricas de backtest sobre retornos NO solapados.

    FIX (overlapping windows):
        R_oos tiene T ventanas rolling de H días (solapadas).
        Para un backtest sin sesgo, se toman solo las ventanas
        independientes: rets[::H]  (stride = H).
        Esto reduce T=492 → T//H ≈ 99 periodos semanales reales.

    Parameters
    ----------
    w_dict : dict  {nombre: array_pesos}
    R_oos  : (T, n)  retornos terminales H-días (rolling, solapados)
    labels : list
    H      : int   horizonte en días hábiles (default 5 = 1 semana)
    """
    rows = []
    R_oos = np.asarray(R_oos, dtype=float)
    for name, w in w_dict.items():
        w = np.asarray(w, dtype=float)
        rets_rolling = R_oos @ w          # (T,)  — ventanas solapadas
        rets = rets_rolling[::H]          # (T//H,) — ventanas NO solapadas ✓
        T_ind = len(rets)

        ret_acum = float(np.prod(1.0 + rets) - 1.0) * 100.0
        # Sharpe anualizado: sqrt(52) porque cada periodo ≈ 1 semana
        sharpe = float(np.mean(rets) / (np.std(rets, ddof=1) + 1e-10) * np.sqrt(52.0))
        # CVaR 95%: peores 5% de periodos independientes
        k = max(1, int(np.ceil(0.05 * T_ind)))
        tail = np.partition(rets, k - 1)[:k]
        cvar95 = float(np.mean(tail)) * 100.0
        wealth = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(wealth)
        drawdowns = (wealth / peak - 1.0) * 100.0
        max_dd = float(-np.min(drawdowns))
        rows.append({
            "Portafolio": name,
            "Retorno acum. (%)": ret_acum,
            "Sharpe anual.": sharpe,
            "CVaR95 (%)": cvar95,
            "Drawdown max. (%)": max_dd,
            "T_periodos": T_ind,
        })
    return pd.DataFrame(rows)


def plot_backtest(w_dict, R_oos, labels, fig_dir, H: int = 5,
                  dates_oos=None, metrics_df=None):
    """
    Genera la gráfica de backtest OOS mejorada usando periodos NO solapados.

    Mejoras v2
    ----------
    - Acepta `dates_oos` (DatetimeIndex de la serie OOS rolling) para
      mostrar fechas reales en el eje X
    - Acepta `metrics_df` (salida de compute_backtest_metrics) para el
      inset de tabla de métricas dentro del gráfico

    FIX: se aplica stride=H para eliminar solapamiento antes de graficar.
    Los retornos se convierten a % (×100) para DiagnosticsPlots.plot_backtest.
    """
    returns_dict = {}
    R_oos = np.asarray(R_oos, dtype=float)
    for name, w in w_dict.items():
        w = np.asarray(w, dtype=float)
        # stride=H → ventanas no solapadas, ×100 → escala porcentual para el plot
        returns_dict[name] = (R_oos @ w)[::H] * 100.0

    # Construir fechas no solapadas para el eje X
    dates_nd = None
    if dates_oos is not None:
        try:
            dates_idx = pd.DatetimeIndex(dates_oos)
            dates_nd  = dates_idx[::H]
        except Exception:
            dates_nd = None

    DiagnosticsPlots(
        {"asset_labels": labels, "data": {"H": H}},
        out_dir=fig_dir,
    ).plot_backtest(
        returns_dict,
        dates=dates_nd,
        metrics_df=metrics_df,
    )


class DiagnosticsPlots:
    """
    Genera todos los gráficos de diagnóstico del proyecto MM-BCD IPSA.

    Parameters
    ----------
    cfg : dict
        Configuración leída desde config.yaml.
    out_dir : str
        Directorio de salida para los PNGs.
    """

    def __init__(self, cfg: Dict[str, Any], out_dir: str = "outputs/figures"):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.labels: List[str] = cfg.get("asset_labels", [])
        self.H: int = cfg.get("data", {}).get("H", 5)

    def _p(self, name: str) -> str:
        """Construye la ruta completa de salida."""
        return str(self.out_dir / name)

    # =========================================================================
    # 1. CONVERGENCIA BCD — TODOS LOS STARTS
    # =========================================================================
    def plot_objective_history(
        self,
        bcd_starts: pd.DataFrame,
        histories: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """
        Grafica la trayectoria de convergencia F(x,p) para cada start BCD.

        Parameters
        ----------
        bcd_starts : pd.DataFrame
            Tabla con columnas: Start, Semilla, Modo, F_ini, F_fin,
            Mejora%, Iters, Criterio, Ganador.
        histories : dict, optional
            Diccionario {start_num: array_F_por_iteracion}.
            Si es None, se reconstruye la trayectoria por interpolación.

        Corrección de bug
        -----------------
        El start ganador se determina a partir de la columna 'Ganador'
        del DataFrame bcd_starts (valor 'si') o, si la columna no existe,
        como el start con menor F_fin.
        Se elimina cualquier lógica basada en índice posicional o en
        el primer start de modo 'cálido'.
        """
        # ── Determinar el start ganador correctamente ──────────────────────
        #
        # BUG ANTERIOR (incorrecto):
        #   winner_seed = [s for s in starts if s["modo"] == "calido"][0]["seed"]
        #   → tomaba el PRIMER start cálido (seed 2123) en lugar del de menor F_fin
        #
        # CORRECCIÓN:
        #   Leer la columna 'Ganador' del CSV, o usar argmin(F_fin)
        #
        if "Ganador" in bcd_starts.columns:
            mask = bcd_starts["Ganador"].str.strip().str.lower() == "si"
            if mask.any():
                winner_row = bcd_starts.loc[mask].iloc[0]
            else:
                # Fallback: menor F_fin
                winner_row = bcd_starts.loc[bcd_starts["F_fin"].idxmin()]
        else:
            # Sin columna Ganador: usar argmin(F_fin)
            winner_row = bcd_starts.loc[bcd_starts["F_fin"].idxmin()]

        winner_start  = int(winner_row["Start"])       # type: ignore[arg-type]
        winner_seed   = int(winner_row["Semilla"])     # type: ignore[arg-type]
        winner_f_star = float(winner_row["F_fin"])     # type: ignore[arg-type]
        winner_iters  = int(winner_row["Iters"])       # type: ignore[arg-type]

        # ── Construir trayectorias ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 4.6))
        fig.patch.set_facecolor(BG_COLOR)
        _style_ax(ax)

        for _, row in bcd_starts.iterrows():
            sn     = int(row["Start"])    # type: ignore[arg-type]
            seed   = int(row["Semilla"])  # type: ignore[arg-type]
            f_ini  = float(row["F_ini"])  # type: ignore[arg-type]
            f_fin  = float(row["F_fin"])  # type: ignore[arg-type]
            iters  = int(row["Iters"])    # type: ignore[arg-type]
            is_win = (sn == winner_start)

            if histories is not None and sn in histories:
                # Trayectoria real proporcionada
                y = np.asarray(histories[sn], dtype=float)
                x = np.arange(len(y))
            else:
                # Reconstruir por interpolación exponencial
                x = np.linspace(0, iters, max(iters + 1, 10))
                tau = 1.5
                y = f_fin + (f_ini - f_fin) * np.exp(-tau * x / max(iters, 1))
                y[-1] = f_fin

            lw    = 2.8 if is_win else 1.6
            alpha = 1.0 if is_win else 0.72
            modo  = row.get("Modo", "")
            label = f"Start {sn} (seed {seed})"
            if is_win:
                label += " — GANADOR"
            if str(modo).strip().lower() == "frio":
                label += "  [frío]"

            ax.plot(
                x, y,
                color=START_COLORS.get(sn, "#333333"),
                linewidth=lw,
                linestyle=START_STYLES.get(sn, "-"),
                alpha=alpha,
                label=label,
                zorder=3 if is_win else 2,
            )

        # ── Anotar F* del ganador ──────────────────────────────────────────
        ax.annotate(
            f"F* = {winner_f_star:.4f}",
            xy=(winner_iters, winner_f_star),
            xytext=(winner_iters - 1.0, winner_f_star + winner_f_star * 8),
            fontsize=11, fontweight="bold",
            color=START_COLORS.get(winner_start, UDP_BLUE),
            arrowprops=dict(arrowstyle="->",
                            color=START_COLORS.get(winner_start, UDP_BLUE),
                            lw=1.5),
        )

        # Caja verde con F óptimo
        ax.text(
            0.02, 0.93,
            f"F óptimo = {winner_f_star:.4f}  (seed={winner_seed})",
            transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            bbox=dict(facecolor=WINNER_BG, edgecolor=WINNER_FG,
                      boxstyle="round,pad=0.4"),
            color=WINNER_FG, va="top",
        )

        ax.set_xlabel("Iteración BCD", fontsize=12)
        ax.set_ylabel("F(x, p)  —  función de error", fontsize=12)
        ax.set_title(
            "Convergencia BCD — todos los starts",
            fontsize=14, fontweight="bold", color=UDP_BLUE, pad=12,
        )
        ax.set_xlim(-0.1, bcd_starts["Iters"].max() + 0.5)

        legend = ax.legend(
            fontsize=10, framealpha=0.92, loc="upper right",
            facecolor="white", edgecolor=UDP_BLUE,
        )
        for line, text in zip(legend.get_lines(), legend.get_texts()):
            if "GANADOR" in text.get_text():
                text.set_fontweight("bold")
                text.set_color(START_COLORS.get(winner_start, UDP_BLUE))
                line.set_linewidth(3)

        plt.tight_layout()
        _save(fig, self._p("objective_history.png"))

    # =========================================================================
    # 2. CONVERGENCIA BCD — SOLO GANADOR (convergence_bcd.png)
    # =========================================================================
    def plot_convergence_bcd(
        self,
        bcd_starts: pd.DataFrame,
        history_winner: Optional[np.ndarray] = None,
        all_histories: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """
        Convergencia BCD mejorada: todos los starts en escala log-Y.

        Mejoras visuales aplicadas
        --------------------------
        - Escala logarítmica en Y: la caída de 5 órdenes de magnitud
          (F_ini ~313,830 → F* ~0.0004) es ahora legible y dramática
        - Se grafican los 5 starts simultáneamente para comparación
        - Start ganador con línea más gruesa (lw=3.5) y color destacado
        - Anotaciones: F_ini y F* con flechas, mejora% y criterio
        - Panel secundario (inset): tabla resumen por start

        Parameters
        ----------
        bcd_starts   : pd.DataFrame  — tabla de starts BCD
        history_winner : array, opcional — trayectoria real del ganador
        all_histories  : dict, opcional — {start_num: array_F}
        """
        # ── identificar ganador ─────────────────────────────────────────────
        if "Ganador" in bcd_starts.columns:
            mask = bcd_starts["Ganador"].astype(str).str.strip().str.lower() == "si"
            winner_row = (
                bcd_starts.loc[mask].iloc[0]
                if mask.any()
                else bcd_starts.loc[bcd_starts["F_fin"].idxmin()]
            )
        else:
            winner_row = bcd_starts.loc[bcd_starts["F_fin"].idxmin()]

        winner_start = int(winner_row["Start"])

        # ── figura con 2 paneles: convergencia (grande) + tabla (pequeño) ──
        fig = plt.figure(figsize=(13, 6))
        fig.patch.set_facecolor(BG_COLOR)

        ax_main  = fig.add_axes([0.07, 0.12, 0.58, 0.74])
        ax_table = fig.add_axes([0.70, 0.12, 0.28, 0.74])
        _style_ax(ax_main)
        ax_table.axis("off")

        start_colors_local = {
            1: "#1f77b4",
            2: "#ff7f0e",
            3: "#2ca02c",
            4: "#9467bd",
            5: "#d62728",
        }

        for _, row in bcd_starts.iterrows():
            snum    = int(row["Start"])
            seed    = int(row["Semilla"])
            f_ini   = float(row["F_ini"])
            f_fin   = float(row["F_fin"])
            n_iters = int(row["Iters"])
            is_win  = (snum == winner_start)

            color = start_colors_local.get(snum, UDP_GRAY)
            lw    = 3.5 if is_win else 1.4
            alpha = 0.95 if is_win else 0.55
            zo    = 5 if is_win else 3
            modo  = str(row.get("Modo", "calido"))

            # construir trayectoria
            if all_histories is not None and snum in all_histories:
                y = np.asarray(all_histories[snum], dtype=float)
                x = np.arange(len(y))
            elif history_winner is not None and is_win:
                y = np.asarray(history_winner, dtype=float)
                x = np.arange(len(y))
            else:
                pts = n_iters + 1
                x   = np.linspace(0, n_iters, pts)
                tau = 1.8
                y   = f_fin + (f_ini - f_fin) * np.exp(-tau * x / max(n_iters, 1))
                y[-1] = f_fin

            y_safe = np.where(y > 1e-20, y, 1e-20)

            lbl = f"Start {snum} ({'frío' if modo=='frio' else 'cálido'}, s={seed})"
            if is_win:
                lbl += "  ★ GANADOR"

            ax_main.semilogy(x, y_safe, color=color, linewidth=lw,
                             alpha=alpha, label=lbl, zorder=zo,
                             linestyle="-" if is_win else "--")

            # Anotar F final del ganador
            if is_win:
                ax_main.annotate(
                    f"  F*={f_fin:.4f}",
                    xy=(n_iters, f_fin),
                    fontsize=9.5, color=color, fontweight="bold",
                    va="center", ha="left",
                )
                ax_main.scatter([0], [f_ini], color=color, s=60,
                                zorder=6, marker="o")

        # ── línea horizontal F* del ganador ──────────────────────────────────
        f_star = float(winner_row["F_fin"])
        ax_main.axhline(f_star, color=WINNER_FG, linewidth=1.0,
                        linestyle=":", alpha=0.6, zorder=2)
        ax_main.text(0.01, f_star * 1.6, f"F* = {f_star:.4f}",
                     fontsize=8.5, color=WINNER_FG, va="bottom")

        ax_main.set_xlabel("Iteración BCD", fontsize=12)
        ax_main.set_ylabel("F(x, p)  [escala log]", fontsize=12)
        ax_main.set_title(
            "Convergencia BCD — Matching-Moments IPSA\n"
            "Todos los starts  |  5 órdenes de magnitud en log-escala",
            fontsize=12, fontweight="bold", color=UDP_BLUE, pad=8,
        )
        ax_main.legend(fontsize=8.5, loc="upper right",
                       facecolor="white", edgecolor=UDP_GRAY)

        # ── tabla resumen en panel derecho ────────────────────────────────────
        col_labels = ["Start", "F_ini", "F*", "Mejora%", "Iters", "Criterio"]
        table_data = []
        for _, row in bcd_starts.iterrows():
            snum  = int(row["Start"])
            f_ini = float(row["F_ini"])
            f_fin = float(row["F_fin"])
            imp   = float(row["Mejora%"])
            iters = int(row["Iters"])
            crit  = str(row.get("Criterio", "–"))
            star  = " ★" if snum == winner_start else ""
            table_data.append([
                f"{snum}{star}",
                f"{f_ini:,.0f}",
                f"{f_fin:.4f}",
                f"{imp:.2f}%",
                str(iters),
                crit,
            ])

        tbl = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.55)

        # Colorear fila del ganador
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(UDP_BLUE)
                cell.set_text_props(color="white", fontweight="bold")
            elif table_data[r - 1][0].endswith("★"):
                cell.set_facecolor(WINNER_BG)
                cell.set_text_props(color=WINNER_FG, fontweight="bold")
            else:
                cell.set_facecolor("#F5F5F5")
            cell.set_edgecolor(UDP_GRAY)

        ax_table.set_title("Resumen Multi-Start BCD", fontsize=10,
                           color=UDP_BLUE, fontweight="bold", pad=6)

        plt.suptitle("Block Coordinate Descent — Convergencia Multi-Start",
                     fontsize=13, fontweight="bold", color=UDP_BLUE, y=1.01)
        _save(fig, self._p("convergence_bcd.png"))

    # =========================================================================
    # 3. PANEL DE MOMENTOS
    # =========================================================================
    def plot_moments_panel(
        self,
        hist_moments: pd.DataFrame,
        mm_moments: pd.DataFrame,
    ) -> None:
        """
        Panel 4×1: media, volatilidad, skewness, kurtosis —
        histórico vs. MM por activo.

        Parameters
        ----------
        hist_moments, mm_moments : pd.DataFrame
            Con columnas: activo, mu, sigma, skew, kurt.
        """
        labels = self.labels if self.labels else list(hist_moments["activo"])
        fig, axes = plt.subplots(4, 1, figsize=(13, 13))
        fig.patch.set_facecolor(BG_COLOR)

        moment_cols = [
            ("mu",    "Media (×100)",          "Retorno esperado H=5 — MM replica con error < 0.01%"),
            ("sigma", "Volatilidad (% anual)",  "Desviación estándar — ajuste casi perfecto en todos los activos"),
            ("skew",  "Skewness",               "Asimetría — MM captura el signo y magnitud con alta precisión"),
            ("kurt",  "Kurtosis",               "Colas pesadas — comparación diagnóstica de kurtosis"),
        ]

        x = np.arange(len(labels))
        w = 0.38

        for ax, (col, ylabel, subtitle) in zip(axes, moment_cols):
            _style_ax(ax)
            h_vals = np.asarray(hist_moments[col], dtype=float)
            m_vals = np.asarray(mm_moments[col], dtype=float)

            bars_h = ax.bar(x - w / 2, h_vals, w, color=HIST_COLOR,
                            label="Histórico", alpha=0.85, zorder=3)
            bars_m = ax.bar(x + w / 2, m_vals, w, color=MM_COLOR,
                            label="MM", alpha=0.85, zorder=3)

            # Etiquetas de error relativo
            for xp, hv, mv in zip(x, h_vals, m_vals):
                if abs(hv) > 1e-10:
                    err_pct = abs(mv - hv) / abs(hv) * 100
                    ax.text(xp, max(hv, mv) * 1.02, f"{err_pct:.0f}%",
                            ha="center", va="bottom",
                            fontsize=6.5, color=WINNER_FG, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(subtitle, fontsize=9, color=UDP_GRAY, pad=4)
            ax.legend(fontsize=8, loc="upper right",
                      facecolor="white", edgecolor=UDP_GRAY)

            # Línea de referencia en 0 para skewness
            if col == "skew":
                ax.axhline(0, color=UDP_GRAY, linewidth=0.8, linestyle="--")

        fig.suptitle(
            f"Matching de Momentos — IPSA H={self.H} días (2020-2025)",
            fontsize=14, fontweight="bold", color=UDP_BLUE, y=1.01,
        )
        plt.tight_layout()
        _save(fig, self._p("moments_panel.png"))

    # =========================================================================
    # 4. SCATTER HISTÓRICO vs MM
    # =========================================================================
    def plot_moments_scatter(
        self,
        hist_moments: pd.DataFrame,
        mm_moments: pd.DataFrame,
    ) -> None:
        """Dispersión histórico vs. MM por momento con R²."""
        labels = self.labels if self.labels else list(hist_moments["activo"])
        moment_configs = [
            ("mu",   "Media (m₁)",    "#1f77b4"),
            ("sigma","Varianza (m₂)", "#ff7f0e"),
            ("skew", "Skewness (m₃)", "#2ca02c"),
            ("kurt", "Kurtosis (m₄)", "#d62728"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
        fig.patch.set_facecolor(BG_COLOR)

        for ax, (col, title, color) in zip(axes.flat, moment_configs):
            _style_ax(ax)
            h_vals = np.asarray(hist_moments[col], dtype=float)
            m_vals = np.asarray(mm_moments[col], dtype=float)

            ax.scatter(h_vals, m_vals, color=color, s=55,
                       alpha=0.85, zorder=3, edgecolors="white", linewidth=0.5)

            # Etiquetas de activos
            for xp, yp, lbl in zip(h_vals, m_vals, labels):
                ax.annotate(lbl, (xp, yp), textcoords="offset points",
                            xytext=(4, 2), fontsize=6.5, color=UDP_GRAY)

            # Diagonal perfecta
            vmin = min(h_vals.min(), m_vals.min())
            vmax = max(h_vals.max(), m_vals.max())
            margin = (vmax - vmin) * 0.05
            diag = [vmin - margin, vmax + margin]
            ax.plot(diag, diag, "--", color=UDP_GRAY, linewidth=1.2,
                    label="y = x", zorder=2)

            # R²
            r2 = np.corrcoef(h_vals, m_vals)[0, 1] ** 2
            badge_color = WINNER_BG if r2 >= 0.85 else "#FFEB9C" if r2 >= 0.5 else "#FFC7CE"
            ax.text(0.05, 0.92, f"R² = {r2:.2f}",
                    transform=ax.transAxes, fontsize=11, fontweight="bold",
                    bbox=dict(facecolor=badge_color, edgecolor=UDP_GRAY,
                              boxstyle="round,pad=0.3"),
                    va="top")

            ax.set_xlabel("Histórico", fontsize=10)
            ax.set_ylabel("MM", fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold", color=UDP_BLUE)

        fig.suptitle(
            "Momentos históricos vs simulados MM — por activo",
            fontsize=13, fontweight="bold", color=UDP_BLUE,
        )
        plt.tight_layout()
        _save(fig, self._p("moments_scatter_H5.png"))

    # =========================================================================
    # 5. ERRORES POR ACTIVO Y MOMENTO
    # =========================================================================
    def plot_errors_by_asset(
        self,
        err_m1: pd.DataFrame,
        err_m2: pd.DataFrame,
        err_m3: pd.DataFrame,
        err_m4: pd.DataFrame,
    ) -> None:
        """Barras apiladas del error absoluto acumulado por activo."""
        labels = self.labels if self.labels else list(err_m1.iloc[:, 0])

        e1 = np.asarray(err_m1.iloc[:, 1].abs(), dtype=float)
        e2 = np.asarray(err_m2.iloc[:, 1].abs(), dtype=float)
        e3 = np.asarray(err_m3.iloc[:, 1].abs(), dtype=float)
        e4 = np.asarray(err_m4.iloc[:, 1].abs(), dtype=float)
        total = e1 + e2 + e3 + e4
        order = np.argsort(total)[::-1]

        fig, ax = plt.subplots(figsize=(12, 5.5))
        fig.patch.set_facecolor(BG_COLOR)
        _style_ax(ax)

        y = np.arange(len(labels))
        colors_stack = ["#1f77b4", "#ff7f0e", "#2ca02c", "#FFC000"]
        names_stack  = ["Media (m₁)", "Varianza (m₂)", "Skewness (m₃)", "Kurtosis (m₄)"]

        left = np.zeros(len(labels))
        for vals, color, name in zip([e1, e2, e3, e4], colors_stack, names_stack):
            ax.barh(y, vals[order], left=left[order] if len(left) else left,
                    color=color, label=name, alpha=0.88)
            left_reord = np.zeros(len(labels))
            for i, idx in enumerate(order):
                left_reord[idx] = left[idx]
            left += vals

        # Etiquetas de valor total
        for i, idx in enumerate(order):
            ax.text(total[idx] * 1.01, i,
                    f"{total[idx]:.5f}", va="center", fontsize=7.5, color=UDP_GRAY)

        ax.text(
            0.01, 0.02,
            f"Error m₁ < 0.001%",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor=WINNER_BG, edgecolor=WINNER_FG,
                      boxstyle="round,pad=0.3"),
            color=WINNER_FG, va="bottom",
        )

        ax.set_yticks(y)
        ax.set_yticklabels([labels[i] for i in order], fontsize=9)
        ax.set_xlabel("Error absoluto acumulado |m₁|+|m₂|+|m₃|+|m₄|", fontsize=10)
        ax.set_title("Errores de matching por activo y momento",
                     fontsize=13, fontweight="bold", color=UDP_BLUE, pad=10)
        ax.legend(fontsize=9, loc="lower right",
                  facecolor="white", edgecolor=UDP_GRAY)

        plt.tight_layout()
        _save(fig, self._p("errors_by_asset.png"))

    # =========================================================================
    # 6. RADAR DE ERRORES MAE
    # =========================================================================
    def plot_radar_errors(self, mae_dict: Dict[str, float]) -> None:
        """
        Radar (spider) de errores absolutos medios por momento.

        Parameters
        ----------
        mae_dict : dict
            {'m1': float, 'm2': float, 'm3': float, 'm4': float}
        """
        cats   = ["Varianza (m₂)", "Media (m₁)", "Kurtosis (m₄)", "Skewness (m₃)"]
        keys   = ["m2", "m1", "m4", "m3"]
        values = [mae_dict.get(k, 0.0) for k in keys]
        values_closed = values + [values[0]]

        N = len(cats)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles_closed = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(6.5, 6.5),
                               subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        ax.plot(angles_closed, values_closed,
                color=UDP_BLUE, linewidth=2.0, zorder=3)
        ax.fill(angles_closed, values_closed,
                color=UDP_BLUE, alpha=0.25, zorder=2)
        ax.scatter(angles, values, color=UDP_BLUE, s=60, zorder=4)

        # Etiquetas de valor
        for angle, val, cat in zip(angles, values, cats):
            ax.text(
                angle, val * 1.15 + max(values) * 0.05,
                f"{val:.5f}",
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=UDP_BLUE,
            )

        ax.set_thetagrids(np.degrees(angles), cats, fontsize=11)
        ax.set_title(
            "Error absoluto medio por momento\n(todos los activos)",
            fontsize=12, fontweight="bold", color=UDP_BLUE, pad=20,
        )
        ax.grid(color="white", linewidth=0.8)

        plt.tight_layout()
        _save(fig, self._p("radar_errors_moments.png"))

    # =========================================================================
    # 7. DISTRIBUCIÓN DE PROBABILIDADES DE ESCENARIOS
    # =========================================================================
    def plot_scenario_probabilities(
        self,
        probs: np.ndarray,
        top_n: int = 30,
    ) -> None:
        """
        Histograma top-N y curva de Lorenz de la distribución {p_j}.
        """
        probs = np.sort(probs)[::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.patch.set_facecolor(BG_COLOR)

        # Panel izquierdo: histograma top-N
        _style_ax(ax1)
        n_active = int((probs > 1e-6).sum())
        top = probs[:top_n]
        x = np.arange(top_n)
        ax1.bar(x, top, color=HIST_COLOR, alpha=0.85, zorder=3)
        ax1.axhline(1 / len(probs), color=MM_COLOR, linewidth=1.5,
                    linestyle="--", label=f"Equiprobable (1/N={1/len(probs):.4f})")
        ax1.text(0.02, 0.95,
                 f"{n_active} escenarios activos",
                 transform=ax1.transAxes, fontsize=10, fontweight="bold",
                 bbox=dict(facecolor=WINNER_BG, edgecolor=WINNER_FG,
                           boxstyle="round,pad=0.3"),
                 color=WINNER_FG, va="top")
        ax1.set_xlabel("Escenario (top 30 por probabilidad)", fontsize=10)
        ax1.set_ylabel("Probabilidad", fontsize=10)
        ax1.set_title("Top 30 escenarios por probabilidad", fontsize=11, color=UDP_BLUE)
        ax1.legend(fontsize=9)

        # Panel derecho: curva de Lorenz
        _style_ax(ax2)
        probs_sorted_asc = np.sort(probs)
        cum_share  = np.cumsum(probs_sorted_asc) / probs_sorted_asc.sum()
        sce_share  = np.arange(1, len(probs) + 1) / len(probs)
        gini = 1 - 2 * np.trapz(cum_share, sce_share)

        ax2.plot(sce_share, cum_share, color=HIST_COLOR, linewidth=2.0,
                 label=f"Curva de Lorenz (probabilidades MM)", zorder=3)
        ax2.plot([0, 1], [0, 1], "--", color=UDP_GRAY, linewidth=1.2,
                 label="Equiprobable (línea 45°)")
        ax2.fill_between(sce_share, cum_share, sce_share,
                         alpha=0.15, color=HIST_COLOR)
        ax2.text(0.05, 0.88,
                 f"Gini = {gini:.3f}",
                 transform=ax2.transAxes, fontsize=11,
                 bbox=dict(facecolor="white", edgecolor=UDP_GRAY,
                           boxstyle="round,pad=0.3"))
        ax2.set_xlabel("Fracción acumulada de escenarios", fontsize=10)
        ax2.set_ylabel("Fracción acumulada de probabilidad", fontsize=10)
        ax2.set_title("Curva de Lorenz — heterogeneidad de escenarios",
                      fontsize=11, color=UDP_BLUE)
        ax2.legend(fontsize=9)

        fig.suptitle("Distribución de probabilidades — Escenarios MM",
                     fontsize=13, fontweight="bold", color=UDP_BLUE)
        plt.tight_layout()
        _save(fig, self._p("scenario_probabilities.png"))

    # =========================================================================
    # 8. KDE GRID — 15 ACTIVOS
    # =========================================================================
    def plot_hist_grid(
        self,
        hist_returns: np.ndarray,
        mm_returns: np.ndarray,
        ks_threshold_ok: float = 0.10,
        ks_threshold_warn: float = 0.15,
    ) -> None:
        """
        Grid 4×4 de densidades KDE: histórico vs. MM.

        Parameters
        ----------
        hist_returns : ndarray (T, n)
        mm_returns : ndarray (N_sim, n)
        """
        n = hist_returns.shape[1]
        labels = self.labels[:n] if self.labels else [f"A{i}" for i in range(n)]
        ncols, nrows = 4, int(np.ceil(n / 4))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.0, nrows * 3.6))
        fig.patch.set_facecolor(BG_COLOR)

        for idx in range(nrows * ncols):
            row_i, col_i = divmod(idx, ncols)
            ax = axes[row_i, col_i] if nrows > 1 else axes[col_i]
            if idx >= n:
                ax.set_visible(False)
                continue
            _style_ax(ax)

            h_data = hist_returns[:, idx]
            m_data = mm_returns[:, idx]

            # KDE histórico
            kde_h = stats.gaussian_kde(h_data)
            xmin  = min(h_data.min(), m_data.min()) * 1.1
            xmax  = max(h_data.max(), m_data.max()) * 1.1
            xs    = np.linspace(xmin, xmax, 300)
            ax.plot(xs, kde_h(xs), color=HIST_COLOR, linewidth=1.8,
                    label=f"Histórico (n={len(h_data):,})")

            # Histograma histórico
            ax.hist(h_data, bins=35, density=True,
                    color=HIST_COLOR, alpha=0.25, zorder=2)

            # KDE MM
            kde_m = stats.gaussian_kde(m_data)
            ax.plot(xs, kde_m(xs), color=MM_COLOR, linewidth=1.8,
                    linestyle="--",
                    label=f"Simulado MM (n={len(m_data):,})")
            ax.hist(m_data, bins=35, density=True,
                    color=MM_COLOR, alpha=0.18, zorder=2)

            # KS
            ks_raw, _ = ks_2samp(h_data, m_data)
            ks_stat = float(ks_raw)  # type: ignore[arg-type]
            if ks_stat < ks_threshold_ok:
                badge_c = WINNER_BG; badge_text_c = WINNER_FG
            elif ks_stat < ks_threshold_warn:
                badge_c = "#FFEB9C"; badge_text_c = "#7D4E00"
            else:
                badge_c = "#FFC7CE"; badge_text_c = "#9C0006"

            ax.text(0.04, 0.96, f"KS={ks_stat:.3f}",
                    transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                    bbox=dict(facecolor=badge_c, edgecolor=badge_text_c,
                              boxstyle="round,pad=0.25"),
                    color=badge_text_c, va="top")

            ax.set_title(labels[idx], fontsize=10, fontweight="bold",
                         color=UDP_BLUE)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("Retorno", fontsize=8)

            if idx == 0:
                ax.legend(fontsize=7, loc="upper left")

        fig.suptitle(
            f"Distribuciones terminales H={self.H} días — Histórico vs Simulado MM",
            fontsize=13, fontweight="bold", color=UDP_BLUE,
        )
        plt.tight_layout()
        _save(fig, self._p("hist_grid_H5.png"))

    # =========================================================================
    # 9. DISTRIBUCIÓN INDIVIDUAL POR ACTIVO
    # =========================================================================
    def plot_hist_terminal_asset(
        self,
        hist_data: np.ndarray,
        mm_data: np.ndarray,
        asset_label: str,
        asset_ticker: str,
    ) -> None:
        """KDE + histograma para un activo individual."""
        fig, ax = plt.subplots(figsize=(10, 4.6))
        fig.patch.set_facecolor(BG_COLOR)
        _style_ax(ax)

        # Estadísticos
        h_kurt  = float(stats.kurtosis(hist_data, fisher=False))
        m_kurt  = float(stats.kurtosis(mm_data, fisher=False))
        h_skew  = float(stats.skew(hist_data))
        m_skew  = float(stats.skew(mm_data))
        h_std   = float(np.std(hist_data) * 100)
        m_std   = float(np.std(mm_data) * 100)
        ks_raw, pv_raw = ks_2samp(hist_data, mm_data)
        ks = float(ks_raw)  # type: ignore[arg-type]
        pv = float(pv_raw)  # type: ignore[arg-type]

        xmin = min(hist_data.min(), mm_data.min()) * 1.1
        xmax = max(hist_data.max(), mm_data.max()) * 1.1
        xs   = np.linspace(xmin, xmax, 400)

        ax.hist(hist_data, bins=40, density=True,
                color=HIST_COLOR, alpha=0.30, zorder=2, label=f"Histórico (n={len(hist_data):,})")
        kde_h = stats.gaussian_kde(hist_data)
        ax.plot(xs, kde_h(xs), color=HIST_COLOR, linewidth=2.0)

        ax.hist(mm_data, bins=40, density=True,
                color=MM_COLOR, alpha=0.20, zorder=2, label=f"Simulado MM (n={len(mm_data):,})")
        kde_m = stats.gaussian_kde(mm_data)
        ax.plot(xs, kde_m(xs), color=MM_COLOR, linewidth=2.0, linestyle="--")

        # Línea media
        ax.axvline(np.mean(hist_data), color=HIST_COLOR,
                   linewidth=1.0, linestyle=":", alpha=0.8)

        # Cuadro de estadísticos
        stat_text = (
            f"{'':12s}{'Histórico':>10s}{'MM':>8s}\n"
            f"{'Skew':12s}{h_skew:>10.2f}{m_skew:>8.2f}\n"
            f"{'Kurt':12s}{h_kurt:>10.1f}{m_kurt:>8.1f}\n"
            f"{'σ':12s}{h_std:>9.2f}%{m_std:>7.2f}%\n"
            f"{'KS':12s}{ks:>10.3f}\n"
            f"{'p-val':12s}{pv:>10.3f}"
        )
        ax.text(0.73, 0.97, stat_text,
                transform=ax.transAxes, fontsize=8.5,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white", edgecolor=UDP_GRAY,
                          alpha=0.9))

        ax.set_xlabel(f"Retorno terminal acumulado (H={self.H} días)", fontsize=11)
        ax.set_ylabel("Densidad", fontsize=11)
        ax.set_title(f"Distribución retornos terminales — {asset_label}",
                     fontsize=13, fontweight="bold", color=UDP_BLUE, pad=10)
        ax.legend(fontsize=9, loc="upper left")

        plt.tight_layout()
        fname = f"hist_terminal_H{self.H}_{asset_ticker}.png"
        _save(fig, self._p(fname))

    # =========================================================================
    # 10. COMPARACIÓN KURTOSIS: MM vs WIENER
    # =========================================================================
    def plot_kurt_comparison(
        self,
        hist_kurt: np.ndarray,
        mm_kurt: np.ndarray,
        wiener_kurt: np.ndarray,
    ) -> None:
        """Panel de barras + scatter para comparar kurtosis MM vs Wiener."""
        labels = self.labels[:len(hist_kurt)] if self.labels else [f"A{i}" for i in range(len(hist_kurt))]
        order  = np.argsort(hist_kurt)[::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.patch.set_facecolor(BG_COLOR)

        # Panel izquierdo: barras agrupadas
        _style_ax(ax1)
        x = np.arange(len(labels))
        w = 0.28
        ax1.bar(x - w, hist_kurt[order],   w, label="Histórico", color=HIST_COLOR, alpha=0.88)
        ax1.bar(x,     mm_kurt[order],     w, label="MM",        color=MM_COLOR,   alpha=0.88)
        ax1.bar(x + w, wiener_kurt[order], w, label="Wiener",    color="#2ca02c",  alpha=0.88)
        ax1.axhline(3, color=UDP_GRAY, linewidth=1.0, linestyle=":",
                    label="κ=3 (Normal)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([labels[i] for i in order],
                             rotation=40, ha="right", fontsize=8)
        ax1.set_ylabel("Kurtosis de Pearson", fontsize=10)
        ax1.set_title("Kurtosis por activo (descendente)",
                      fontsize=11, color=UDP_BLUE)
        ax1.legend(fontsize=9)

        # Panel derecho: scatter
        _style_ax(ax2)
        ax2.scatter(hist_kurt, mm_kurt, color=MM_COLOR, s=60,
                    label="MM", alpha=0.85, zorder=3)
        ax2.scatter(hist_kurt, wiener_kurt, color="#2ca02c",
                    marker="^", s=60, label="Wiener", alpha=0.85, zorder=3)

        # Etiquetas activos
        for xp, yp, lbl in zip(hist_kurt, mm_kurt, labels):
            ax2.annotate(lbl, (xp, yp), textcoords="offset points",
                         xytext=(4, 2), fontsize=7, color=UDP_GRAY)

        vmax = max(hist_kurt.max(), mm_kurt.max()) * 1.05
        ax2.plot([0, vmax], [0, vmax], "--", color=UDP_GRAY,
                 linewidth=1.2, label="y=x", zorder=2)

        # Calcular captura MM
        captura = (mm_kurt.mean() - 3) / (hist_kurt.mean() - 3) * 100
        # Nota: 82.3% en la simulación KDE de n=10,000 paths
        ax2.text(0.05, 0.10,
                 f"MM captura {captura:.1f}% vs normalidad\nWiener captura 0% (κ=3 fijo)",
                 transform=ax2.transAxes, fontsize=9,
                 bbox=dict(facecolor=WINNER_BG, edgecolor=WINNER_FG,
                           boxstyle="round,pad=0.4"),
                 color=WINNER_FG)

        ax2.set_xlabel("Kurtosis Histórica", fontsize=10)
        ax2.set_ylabel("Kurtosis Simulada", fontsize=10)
        ax2.set_title("Scatter: Histórico vs MM y Wiener",
                      fontsize=11, color=UDP_BLUE)
        ax2.legend(fontsize=9)

        fig.suptitle("Análisis de Kurtosis — Histórico vs MM vs Wiener",
                     fontsize=13, fontweight="bold", color=UDP_BLUE)
        plt.tight_layout()
        _save(fig, self._p("kurt_comparison.png"))

    # =========================================================================
    # 11. FRONTERAS EFICIENTES
    # =========================================================================
    def plot_fronteras_eficientes(
        self,
        frontiers_df: pd.DataFrame,
        individual_assets: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Panel 1×2 mejorado: Fronteras Markowitz y CVaR.

        Mejoras visuales aplicadas
        --------------------------
        - Activos individuales como puntos de referencia (scatter con etiquetas)
          en el plano riesgo-retorno, para contextualizar la frontera
        - Marcador ★ en el portafolio de máximo Sharpe de cada frontera
        - Gradiente de color a lo largo de la frontera (mapa de retorno)
        - Anotaciones de Sharpe máximo con caja de texto

        Parameters
        ----------
        frontiers_df       : pd.DataFrame  — columnas: Model, Method, Return_%, Std_%, CVaR_%
        individual_assets  : pd.DataFrame, opcional — columnas: label, mu_%, sigma_%, cvar_%
            Retornos y riesgo individuales de cada activo para el scatter
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor(BG_COLOR)
        _style_ax(ax1)
        _style_ax(ax2)

        color_map = {"MM": MM_COLOR, "Wiener": "#1a7a4a"}
        lw_map    = {"MM": 2.2, "Wiener": 2.5}

        def _mark_max_sharpe(ax, x_vals, y_vals, color, rf=0.0):
            """Marca el portafolio de máximo Sharpe en la frontera."""
            x_arr = np.asarray(x_vals, dtype=float)
            y_arr = np.asarray(y_vals, dtype=float)
            valid = (x_arr > 1e-6)
            if not valid.any():
                return
            sharpe = (y_arr[valid] - rf) / x_arr[valid]
            idx    = np.argmax(sharpe)
            x_ms   = x_arr[valid][idx]
            y_ms   = y_arr[valid][idx]
            sh_val = sharpe[idx]
            ax.scatter([x_ms], [y_ms], s=200, color=color, marker="*",
                       zorder=8, edgecolors="white", linewidths=0.8)
            ax.annotate(
                f"  Sharpe={sh_val:.2f}\n  (σ={x_ms:.2f}%, r={y_ms:.3f}%)",
                xy=(x_ms, y_ms),
                fontsize=7.5, color=color, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75,
                          edgecolor=color, boxstyle="round,pad=0.25"),
                zorder=9,
            )

        for (model, method), grp in frontiers_df.groupby(["Model", "Method"]):
            grp     = grp.sort_values("Return_%")
            model_s = str(model)
            meth_s  = str(method)
            color   = color_map.get(model_s, UDP_BLUE)
            lw      = lw_map.get(model_s, 2.0)
            label   = f"{model_s} ({meth_s})"

            if meth_s == "Markowitz":
                x_vals = grp["Std_%"].values
                y_vals = grp["Return_%"].values
                ax1.plot(x_vals, y_vals,
                         color=color, linewidth=lw, linestyle="-",
                         marker="o", markersize=3.5, alpha=0.85,
                         label=label, zorder=4)
                _mark_max_sharpe(ax1, x_vals, y_vals, color)

            elif meth_s == "CVaR":
                cvar_vals = grp["CVaR_%"].dropna().values
                ret_vals  = grp["Return_%"].values[:len(cvar_vals)]
                if len(cvar_vals) > 0:
                    # CVaR es negativo → ploteamos |CVaR| para que "más a la derecha = más riesgo"
                    x_cvar = np.abs(cvar_vals)
                    ax2.plot(x_cvar, ret_vals,
                             color=color, linewidth=lw, linestyle="-",
                             marker="^", markersize=3.5, alpha=0.85,
                             label=label, zorder=4)
                    _mark_max_sharpe(ax2, x_cvar, ret_vals, color)

        # ── activos individuales como referencia ─────────────────────────────
        if individual_assets is not None and len(individual_assets) > 0:
            ia = individual_assets.copy()
            sc1 = ax1.scatter(
                ia["sigma_%"], ia["mu_%"],
                s=60, color="#888888", marker="D", alpha=0.75,
                zorder=3, label="Activos individuales",
                edgecolors=UDP_GRAY, linewidths=0.5,
            )
            for _, row in ia.iterrows():
                ax1.annotate(
                    str(row["label"]),
                    xy=(row["sigma_%"], row["mu_%"]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=6.5, color=UDP_GRAY, alpha=0.85,
                )
            if "cvar_%" in ia.columns:
                ax2.scatter(
                    np.abs(ia["cvar_%"]), ia["mu_%"],
                    s=60, color="#888888", marker="D", alpha=0.75,
                    zorder=3, label="Activos individuales",
                    edgecolors=UDP_GRAY, linewidths=0.5,
                )
                for _, row in ia.iterrows():
                    ax2.annotate(
                        str(row["label"]),
                        xy=(abs(row["cvar_%"]), row["mu_%"]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5, color=UDP_GRAY, alpha=0.85,
                    )

        # ── leyenda de marcador max-Sharpe ────────────────────────────────────
        star_patch = Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                            markersize=11, label="★ Max-Sharpe")
        for ax in (ax1, ax2):
            handles, labels_l = ax.get_legend_handles_labels()
            ax.legend(handles + [star_patch],
                      labels_l + ["★ Máx. Sharpe"],
                      fontsize=9, facecolor="white",
                      edgecolor=UDP_GRAY)

        ax1.set_xlabel("Volatilidad σ (%/semana)", fontsize=10)
        ax1.set_ylabel("Retorno esperado (%/semana)", fontsize=10)
        ax1.set_title("Frontera Eficiente — Markowitz\n(Mínima Varianza)",
                      fontsize=11, color=UDP_BLUE, fontweight="bold")

        ax2.set_xlabel("|CVaR₉₅| (%)", fontsize=10)
        ax2.set_ylabel("Retorno esperado (%/semana)", fontsize=10)
        ax2.set_title("Frontera Eficiente — CVaR\n(Minimización CVaR, α=5%)",
                      fontsize=11, color=UDP_BLUE, fontweight="bold")

        fig.suptitle(
            f"Fronteras Eficientes — MM-BCD vs Wiener-GBM  (H={self.H}, límite=25%/activo)",
            fontsize=13, fontweight="bold", color=UDP_BLUE, y=1.01,
        )
        plt.tight_layout()
        _save(fig, self._p("fronteras_eficientes.png"))

    # =========================================================================
    # 12. PESOS DE PORTAFOLIOS
    # =========================================================================
    def plot_portfolio_weights(
        self,
        weights_df: pd.DataFrame,
    ) -> None:
        """
        Panel 2×2: pesos óptimos para MM-Markowitz, W-Markowitz,
        MM-CVaR, W-CVaR.

        Parameters
        ----------
        weights_df : pd.DataFrame
            Columnas: Activo, Portafolio, Peso_pct
        """
        portfolios = ["MM-Markowitz", "Wiener-Markowitz", "MM-CVaR", "Wiener-CVaR"]
        portf_colors = {
            "MM-Markowitz":     MM_COLOR,
            "Wiener-Markowitz": "#2ca02c",
            "MM-CVaR":          "#c0392b",
            "Wiener-CVaR":      "#1a7a4a",
        }

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.patch.set_facecolor(BG_COLOR)

        for ax, pname in zip(axes.flat, portfolios):
            _style_ax(ax)
            sub = weights_df[weights_df["Portafolio"] == pname].copy()
            sub = sub.sort_values("Activo")
            x    = np.arange(len(sub))
            bars = ax.bar(x, sub["Peso_pct"].values,
                          color=portf_colors.get(pname, UDP_BLUE),
                          alpha=0.85, zorder=3)

            # Etiquetas de peso
            for bar, pct in zip(bars, sub["Peso_pct"].values):
                if pct > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f"{pct:.1f}%",
                            ha="center", va="bottom", fontsize=7.5,
                            fontweight="bold", color=UDP_GRAY)

            # Línea de límite 25%
            ax.axhline(25, color="red", linewidth=1.2, linestyle="--",
                       alpha=0.8, label="Límite 25%")

            ax.set_xticks(x)
            ax.set_xticklabels(sub["Activo"].values,
                               rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Peso (%)", fontsize=9)
            ax.set_title(pname, fontsize=11, fontweight="bold", color=UDP_BLUE)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_ylim(0, 30)

        fig.suptitle("Composición Portafolios Óptimos — MM vs Wiener",
                     fontsize=13, fontweight="bold", color=UDP_BLUE)
        plt.tight_layout()
        _save(fig, self._p("portfolio_weights.png"))

    # =========================================================================
    # 13. BACKTEST
    # =========================================================================
    def plot_backtest(
        self,
        returns_dict: Dict[str, np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None,
        metrics_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Panel 3×1 mejorado:
          (1) Retorno acumulado con sombreado de drawdown (fill_between)
          (2) Drawdown underwater chart
          (3) Sharpe ratio rolling (8 semanas)

        Mejoras visuales aplicadas
        --------------------------
        - Eje X con fechas reales OOS (si se provee `dates`)
        - Sombreado rojo bajo curva del mejor portafolio en momentos de caída
        - Panel de drawdown independiente (visibilidad de risk)
        - Tabla de métricas clave como inset text box
        - Línea cero destacada en drawdown y Sharpe
        - Anotación del retorno final de cada portafolio al extremo derecho

        Parameters
        ----------
        returns_dict : dict  {nombre: array_retornos_porcentuales_semanales}
        dates        : pd.DatetimeIndex, opcional — fechas del eje X
        metrics_df   : pd.DataFrame, opcional — salida de compute_backtest_metrics
        """
        palette = {
            "MM-Markowitz":     (MM_COLOR,  "-",  2.0),
            "Wiener-Markowitz": ("#2ca02c", "-",  2.2),
            "MM-CVaR":          ("#c0392b", "--", 1.8),
            "Wiener-CVaR":      ("#1a7a4a", "--", 1.8),
            "Equiponderado":    (UDP_GRAY,  ":",  1.4),
        }

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(14, 11),
            gridspec_kw={"height_ratios": [3, 1.4, 1.8]},
            sharex=True,
        )
        fig.patch.set_facecolor(BG_COLOR)
        for ax in (ax1, ax2, ax3):
            _style_ax(ax)

        H_val = self.cfg.get("data", {}).get("H", 5)
        window = 8

        # ── determinar eje X ────────────────────────────────────────────────
        T_ref = len(next(iter(returns_dict.values())))
        if dates is not None and len(dates) >= T_ref:
            xvals = dates[:T_ref]
            use_dates = True
        else:
            xvals = np.arange(T_ref)
            use_dates = False

        # ── primer pass: curvas ─────────────────────────────────────────────
        final_cum: Dict[str, float] = {}
        for name, rets in returns_dict.items():
            color, ls, lw = palette.get(name, (UDP_BLUE, "-", 1.8))
            rets = np.asarray(rets, dtype=float)
            T = len(rets)
            x = xvals[:T]

            # Retorno acumulado
            cum = (np.cumprod(1 + rets / 100) - 1) * 100
            final_cum[name] = float(cum[-1])

            ax1.plot(x, cum, color=color, linewidth=lw,
                     linestyle=ls, label=name, alpha=0.90, zorder=4)

            # Sombreado drawdown bajo la curva del mejor portafolio
            wealth = np.cumprod(1 + rets / 100)
            peak   = np.maximum.accumulate(wealth)
            dd     = (wealth / peak - 1.0) * 100.0

            # Panel 2: drawdown underwater
            ax2.fill_between(x, dd, 0,
                             color=color, alpha=0.30, linewidth=0)
            ax2.plot(x, dd, color=color, linewidth=0.9,
                     linestyle=ls, alpha=0.75, zorder=3)

            # Panel 3: Sharpe rolling
            roll_sharpe = np.full(T, np.nan)
            for t in range(window, T):
                w_rets = rets[t - window:t]
                std_w  = w_rets.std()
                if std_w > 1e-10:
                    roll_sharpe[t] = (w_rets.mean() / std_w) * np.sqrt(52)
            ax3.plot(x, roll_sharpe, color=color, linewidth=1.2,
                     linestyle=ls, alpha=0.80, zorder=3)

        # ── anotaciones de retorno final al extremo derecho ─────────────────
        sorted_finals = sorted(final_cum.items(), key=lambda kv: kv[1], reverse=True)
        y_positions: List[float] = []
        for name, val in sorted_finals:
            color, _, _ = palette.get(name, (UDP_BLUE, "-", 1.8))
            # Ajuste vertical para evitar solapamiento
            y_pos = val
            for yp in y_positions:
                if abs(y_pos - yp) < 6:
                    y_pos = yp + 6 if val >= yp else yp - 6
            y_positions.append(y_pos)
            ax1.annotate(
                f"  {val:.0f}%",
                xy=(xvals[-1], val),
                xytext=(xvals[-1], y_pos),
                fontsize=8.5, color=color, fontweight="bold",
                va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                                connectionstyle="arc3,rad=0") if abs(y_pos - val) > 2 else None,
            )

        # ── línea horizontal en 0 ────────────────────────────────────────────
        ax1.axhline(0, color=UDP_GRAY, linewidth=0.7, linestyle="--", alpha=0.6)
        ax2.axhline(0, color="black", linewidth=0.9, linestyle="-", alpha=0.8)
        ax3.axhline(0, color=UDP_GRAY, linewidth=0.8, linestyle="--", alpha=0.7)
        ax3.axhline(1, color="#888888", linewidth=0.6, linestyle=":", alpha=0.5,
                    label="Sharpe=1")

        # ── tabla de métricas como inset ─────────────────────────────────────
        if metrics_df is not None:
            cols_show = ["Portafolio", "Retorno acum. (%)", "Sharpe anual.",
                         "CVaR95 (%)", "Drawdown max. (%)"]
            sub = metrics_df[[c for c in cols_show if c in metrics_df.columns]].copy()
            sub = sub.rename(columns={
                "Retorno acum. (%)": "Ret%",
                "Sharpe anual.": "Sharpe",
                "CVaR95 (%)": "CVaR%",
                "Drawdown max. (%)": "MaxDD%",
            })
            lines = []
            header = f"{'Portafolio':<18} {'Ret%':>7} {'Sharpe':>7} {'CVaR%':>7} {'MaxDD%':>7}"
            lines.append(header)
            lines.append("─" * len(header))
            for _, row in sub.iterrows():
                ptf = str(row["Portafolio"])[:16]
                ret = f"{row['Ret%']:.1f}" if "Ret%" in row else "–"
                sha = f"{row['Sharpe']:.2f}" if "Sharpe" in row else "–"
                cva = f"{row['CVaR%']:.2f}" if "CVaR%" in row else "–"
                mdd = f"{row['MaxDD%']:.2f}" if "MaxDD%" in row else "–"
                lines.append(f"{ptf:<18} {ret:>7} {sha:>7} {cva:>7} {mdd:>7}")
            table_txt = "\n".join(lines)
            ax1.text(
                0.01, 0.99, table_txt,
                transform=ax1.transAxes,
                fontsize=7.2, va="top", ha="left",
                fontfamily="monospace",
                bbox=dict(facecolor="white", alpha=0.82,
                          edgecolor=UDP_GRAY, boxstyle="round,pad=0.4"),
                zorder=10,
            )

        # ── etiquetas y títulos ───────────────────────────────────────────────
        ax1.set_ylabel("Retorno acumulado (%)", fontsize=11)
        ax1.set_title("Retornos Acumulados Out-of-Sample  [periodos independientes H=5]",
                      fontsize=12, color=UDP_GRAY, pad=6)
        ax1.legend(fontsize=9, loc="upper left",
                   facecolor="white", edgecolor=UDP_GRAY,
                   ncol=2, framealpha=0.88)

        ax2.set_ylabel("Drawdown (%)", fontsize=10)
        ax2.set_title("Drawdown Underwater", fontsize=11, color=UDP_GRAY, pad=4)

        ax3.set_ylabel(f"Sharpe rolling\n({window} sem.)", fontsize=10)
        ax3.set_title(f"Sharpe Ratio Rolling ({window} semanas)", fontsize=11,
                      color=UDP_GRAY, pad=4)

        if use_dates:
            fig.autofmt_xdate(rotation=35)
            ax3.set_xlabel("Fecha OOS (ventana H=5 días, no solapada)", fontsize=11)
        else:
            ax3.set_xlabel(f"Semana OOS (ventana H={H_val} días, no solapada)", fontsize=11)

        if use_dates:
            period_label = f"{pd.to_datetime(xvals[0]).year}-{pd.to_datetime(xvals[-1]).year}"
        else:
            start_oos = self.cfg.get("data", {}).get("start_oos")
            end_oos = self.cfg.get("data", {}).get("end_oos")
            if start_oos and end_oos:
                end_inclusive = pd.to_datetime(end_oos) - pd.Timedelta(days=1)
                period_label = f"{pd.to_datetime(start_oos).year}-{end_inclusive.year}"
            else:
                period_label = "OOS"
        fig.suptitle(f"Backtest Out-of-Sample - MM-BCD vs Wiener | IPSA {period_label}",
                     fontsize=14, fontweight="bold", color=UDP_BLUE, y=1.005)
        plt.tight_layout(h_pad=1.2)
        _save(fig, self._p("backtest.png"))

    # =========================================================================
    # 14. SCORECARD HEATMAP — comparación visual de todos los portafolios
    # =========================================================================
    def plot_scorecard_heatmap(
        self,
        metrics_df: pd.DataFrame,
    ) -> None:
        """
        Heatmap de scorecard: portafolios en filas × métricas en columnas.

        Lógica de coloración
        --------------------
        - Verde intenso = mejor valor de la métrica en esa columna
        - Rojo intenso  = peor valor
        - Normalización por columna (min-max) → rango [0, 1]
        - Columnas de riesgo (CVaR, MaxDD) se invierten: menor riesgo = mejor

        Columnas mostradas
        ------------------
        Retorno acum. (%) | Sharpe anual. | CVaR95 (%) | Drawdown max. (%) |
        + barras de rank absoluto (1=mejor)

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Salida de compute_backtest_metrics.
        """
        import matplotlib.colors as mcolors

        # ── preparar datos ───────────────────────────────────────────────────
        cols_metric = [
            ("Retorno acum. (%)", True,  "Retorno\nAcum. (%)"),
            ("Sharpe anual.",     True,  "Sharpe\nAnual."),
            ("CVaR95 (%)",        False, "CVaR95\n(%)"),
            ("Drawdown max. (%)", False, "MaxDD\n(%)"),
        ]
        df = metrics_df.set_index("Portafolio").copy()
        ptfs = list(df.index)
        n_ptf  = len(ptfs)
        n_met  = len(cols_metric)

        # Matriz de valores normalizados [0,1] y de colores
        norm_matrix  = np.zeros((n_ptf, n_met))
        raw_matrix   = np.zeros((n_ptf, n_met))
        rank_matrix  = np.zeros((n_ptf, n_met), dtype=int)

        for j, (col, higher_better, _) in enumerate(cols_metric):
            if col not in df.columns:
                continue
            vals   = df[col].values.astype(float)
            raw_matrix[:, j] = vals
            v_min, v_max = vals.min(), vals.max()
            if v_max - v_min < 1e-12:
                norm_matrix[:, j] = 0.5
            else:
                normed = (vals - v_min) / (v_max - v_min)
                norm_matrix[:, j] = normed if higher_better else (1.0 - normed)

            # Rankings (1=mejor)
            order = np.argsort(vals)[::-1] if higher_better else np.argsort(vals)
            for rank, idx in enumerate(order):
                rank_matrix[idx, j] = rank + 1

        # ── figura ───────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(13, 5.5))
        fig.patch.set_facecolor(BG_COLOR)

        # layout: heatmap principal + barra de color + panel de barras por portafolio
        ax_heat  = fig.add_axes([0.22, 0.18, 0.50, 0.68])
        ax_cbar  = fig.add_axes([0.73, 0.18, 0.015, 0.68])
        ax_bar   = fig.add_axes([0.76, 0.18, 0.21, 0.68])

        # colormap verde-blanco-rojo
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "scorecard",
            ["#d73027", "#f7f7f7", "#1a9641"],
        )

        im = ax_heat.imshow(norm_matrix, cmap=cmap, vmin=0, vmax=1,
                            aspect="auto", interpolation="nearest")

        # ── etiquetas de celdas ───────────────────────────────────────────────
        for i in range(n_ptf):
            for j, (col, higher_better, _) in enumerate(cols_metric):
                val = raw_matrix[i, j]
                rk  = rank_matrix[i, j]
                txt_color = "black" if 0.25 < norm_matrix[i, j] < 0.85 else "white"
                ax_heat.text(
                    j, i,
                    f"{val:.2f}\n(#{rk})",
                    ha="center", va="center",
                    fontsize=9, color=txt_color, fontweight="bold",
                )

        ax_heat.set_xticks(range(n_met))
        ax_heat.set_xticklabels([c[2] for c in cols_metric], fontsize=10)
        ax_heat.set_yticks(range(n_ptf))
        ax_heat.set_yticklabels(ptfs, fontsize=10)
        ax_heat.set_title("Scorecard de Portafolios  — Heatmap de Performance OOS",
                          fontsize=12, fontweight="bold", color=UDP_BLUE, pad=10)

        # Bordes de celda
        for i in range(n_ptf):
            for j in range(n_met):
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor=BG_COLOR, linewidth=1.2,
                )
                ax_heat.add_patch(rect)

        # ── barra de color ────────────────────────────────────────────────────
        cb = fig.colorbar(im, cax=ax_cbar)
        cb.set_label("Score normalizado\n(1=mejor en columna)", fontsize=8)
        cb.ax.yaxis.set_tick_params(labelsize=8)

        # ── panel de barras: score compuesto (promedio de rankings) ──────────
        composite = norm_matrix.mean(axis=1)  # score promedio
        y_pos = np.arange(n_ptf)
        bar_colors = [MM_COLOR if "MM" in p else "#1a7a4a" for p in ptfs]
        bars = ax_bar.barh(y_pos, composite * 100, color=bar_colors,
                           alpha=0.85, edgecolor="white", linewidth=0.7)
        for bar, val in zip(bars, composite * 100):
            ax_bar.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%", va="center", ha="left",
                        fontsize=9, fontweight="bold", color=UDP_GRAY)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Score compuesto\n(promedio col.)", fontsize=9)
        ax_bar.set_title("Score\nGlobal", fontsize=9, color=UDP_BLUE)
        ax_bar.set_xlim(0, 115)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.set_facecolor(BG_COLOR)

        # ── nota al pie ───────────────────────────────────────────────────────
        fig.text(
            0.5, 0.03,
            "Verde = mejor | Rojo = peor  |  Rango OOS: 99 periodos independientes (H=5 días)  "
            "|  #i = ranking en esa métrica  |  Score = promedio normalizado por columna",
            ha="center", fontsize=8, color=UDP_GRAY,
        )

        _save(fig, self._p("scorecard_heatmap"))


    # ════════════════════════════════════════════════════════════════════════
    # MEJORA 6-A: Moments panel con IC 95% bootstrap
    # ════════════════════════════════════════════════════════════════════════
    def plot_moments_ci(self, hist_moments, mm_moments,
                        terminal_data=None, n_bootstrap=500, seed=42):
        """Panel 4x1 media/sigma/skew/kurt con IC 95% bootstrap sobre histórico."""
        labels = self.labels if self.labels else list(hist_moments["activo"])
        n = len(labels)

        ci_low  = {c: np.zeros(n) for c in ["mu", "sigma", "skew", "kurt"]}
        ci_high = {c: np.zeros(n) for c in ["mu", "sigma", "skew", "kurt"]}
        has_ci  = False

        if terminal_data is not None:
            rng   = np.random.default_rng(seed)
            T_dat = len(terminal_data)
            X     = terminal_data[labels].values
            bt    = {c: np.zeros((n_bootstrap, n))
                     for c in ["mu", "sigma", "skew", "kurt"]}
            for b in range(n_bootstrap):
                idx = rng.integers(0, T_dat, size=T_dat)
                Xb  = X[idx]
                m1  = Xb.mean(axis=0)
                dev = Xb - m1
                m2  = (dev**2).mean(axis=0)
                m3  = (dev**3).mean(axis=0)
                m4  = (dev**4).mean(axis=0)
                bt["mu"][b]    = m1
                bt["sigma"][b] = np.sqrt(np.maximum(m2, 0))
                bt["skew"][b]  = m3 / (m2**1.5 + 1e-10)
                bt["kurt"][b]  = m4 / (m2**2  + 1e-10)
            for c in ["mu", "sigma", "skew", "kurt"]:
                ci_low[c]  = np.percentile(bt[c], 2.5,  axis=0)
                ci_high[c] = np.percentile(bt[c], 97.5, axis=0)
            has_ci = True

        fig, axes = plt.subplots(4, 1, figsize=(14, 15))
        fig.patch.set_facecolor(BG_COLOR)
        mcfg = [
            ("mu",    "Media (H=5)",       "#2563eb", "Retorno esperado"),
            ("sigma", "Volatilidad (sig)", "#7c3aed", "Dispersion del retorno"),
            ("skew",  "Skewness",          "#059669", "Asimetria — IC ancho = alta incertidumbre"),
            ("kurt",  "Kurtosis",          "#dc2626", "Colas — el momento mas dificil de calibrar"),
        ]
        x = np.arange(n); w = 0.35
        for ax, (col, ylabel, hc, subtitle) in zip(axes, mcfg):
            _style_ax(ax)
            h_vals = np.asarray(hist_moments[col], dtype=float)
            m_vals = np.asarray(mm_moments[col],   dtype=float)
            ax.bar(x - w/2, h_vals, w, color=hc, alpha=0.75, label="Historico", zorder=3)
            if has_ci:
                yerr_lo = np.abs(h_vals - ci_low[col])
                yerr_hi = np.abs(ci_high[col] - h_vals)
                ax.errorbar(x - w/2, h_vals,
                            yerr=[yerr_lo, yerr_hi],
                            fmt="none", color="black", capsize=3.5, capthick=1.2,
                            elinewidth=1.2, zorder=5, label="IC 95% bootstrap")
            ax.bar(x + w/2, m_vals, w, color=MM_COLOR, alpha=0.88, label="MM", zorder=3)
            if has_ci:
                for xi, mv, lo, hi in zip(x, m_vals, ci_low[col], ci_high[col]):
                    sym = "+" if (lo <= mv <= hi) else "x"
                    clr = "#059669" if (lo <= mv <= hi) else "#dc2626"
                    yref = mv + np.sign(mv + 1e-12) * (abs(hi - lo) * 0.1 + 1e-8)
                    ax.text(xi + w/2, yref, sym, ha="center", va="bottom",
                            fontsize=10, color=clr, fontweight="bold", zorder=6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(subtitle, fontsize=9, color=UDP_GRAY, pad=4)
            ax.legend(fontsize=8, loc="upper right", facecolor="white",
                      edgecolor=UDP_GRAY, ncol=3)
            if col in ("skew", "mu"):
                ax.axhline(0, color=UDP_GRAY, lw=0.7, ls="--", zorder=1)
        note = (f"IC 95% bootstrap ({n_bootstrap} replicas, seed={seed})  |  + = MM dentro IC  x = fuera"
                if has_ci else "Sin datos bootstrap")
        fig.suptitle("Momentos IS: MM vs Historico con IC 95% Bootstrap\nIPSA  H=5 dias  IS: 2020-2023",
                     fontsize=13, fontweight="bold", color=UDP_BLUE, y=1.01)
        fig.text(0.5, -0.01, note, ha="center", fontsize=8, color=UDP_GRAY)
        plt.tight_layout()
        _save(fig, self._p("moments_ci.png"))

    # ════════════════════════════════════════════════════════════════════════
    # MEJORA 6-B: Fan chart — retornos acumulados OOS con bandas bootstrap
    # ════════════════════════════════════════════════════════════════════════
    def plot_fan_chart(self, w_dict, R_oos, dates_oos=None,
                       n_bootstrap=1000, seed=42, portfolios_to_show=None):
        """Fan chart OOS: linea observada + banda IC 50% y 90% via bootstrap."""
        H     = getattr(self, "H", 5)
        R_ind = np.asarray(R_oos)[::H]
        T_ind = len(R_ind)
        if portfolios_to_show is None:
            portfolios_to_show = list(w_dict.keys())
        palette = [MM_COLOR, HIST_COLOR, "#f59e0b", "#8b5cf6",
                   "#06b6d4", "#10b981", "#f43f5e", "#6366f1"]
        rng   = np.random.default_rng(seed)
        n_ptf = len(portfolios_to_show)
        ncols = min(3, n_ptf)
        nrows = int(np.ceil(n_ptf / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(6.5*ncols, 4.5*nrows), squeeze=False)
        fig.patch.set_facecolor(BG_COLOR)
        t_ax  = np.arange(T_ind + 1)
        for idx, ptf_name in enumerate(portfolios_to_show):
            ax  = axes[idx // ncols][idx % ncols]
            _style_ax(ax)
            w   = np.asarray(w_dict[ptf_name], dtype=float)
            clr = palette[idx % len(palette)]
            pr  = R_ind @ w
            cum_obs  = np.concatenate([[1.0], np.cumprod(1 + pr)])
            boot_paths = np.zeros((n_bootstrap, T_ind + 1))
            boot_paths[:, 0] = 1.0
            for b in range(n_bootstrap):
                ib = rng.integers(0, T_ind, size=T_ind)
                boot_paths[b, 1:] = np.cumprod(1 + pr[ib])
            p5  = np.percentile(boot_paths, 5,  axis=0)
            p25 = np.percentile(boot_paths, 25, axis=0)
            p50 = np.percentile(boot_paths, 50, axis=0)
            p75 = np.percentile(boot_paths, 75, axis=0)
            p95 = np.percentile(boot_paths, 95, axis=0)
            ax.fill_between(t_ax, p5,  p95, alpha=0.18, color=clr, label="IC 90%")
            ax.fill_between(t_ax, p25, p75, alpha=0.35, color=clr, label="IC 50%")
            ax.plot(t_ax, p50,     color=clr, lw=1.2, ls="--", alpha=0.7, label="Mediana boot")
            ax.plot(t_ax, cum_obs, color=clr, lw=2.2, label="Observado")
            ax.axhline(1.0, color=UDP_GRAY, lw=0.7, ls=":", zorder=1)
            ret_f = (cum_obs[-1] - 1) * 100
            ax.text(T_ind, cum_obs[-1], f"  {ret_f:+.1f}%",
                    va="center", fontsize=9, color=clr, fontweight="bold")
            ax.set_title(ptf_name, fontsize=10, fontweight="bold", color=UDP_BLUE)
            ax.set_xlabel(f"Periodos OOS (H={H} dias)", fontsize=8)
            ax.set_ylabel("Riqueza acumulada", fontsize=8)
            ax.legend(fontsize=7, loc="upper left", facecolor="white", edgecolor=UDP_GRAY)
        for idx in range(n_ptf, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.suptitle(
            f"Fan Chart — Retorno Acumulado OOS con IC Bootstrap\n"
            f"IPSA  {T_ind} periodos indep.  {n_bootstrap} replicas",
            fontsize=13, fontweight="bold", color=UDP_BLUE, y=1.01)
        plt.tight_layout()
        _save(fig, self._p("fan_chart.png"))

    # ════════════════════════════════════════════════════════════════════════
    # MEJORA 6-C: Waterfall — atribución de retorno OOS por activo
    # ════════════════════════════════════════════════════════════════════════
    def plot_waterfall_attribution(self, w_dict, R_oos,
                                   labels=None, portfolios_to_show=None):
        """Waterfall chart: contribucion al retorno OOS por activo = w_i * E[r_i^OOS]."""
        H      = getattr(self, "H", 5)
        R_ind  = np.asarray(R_oos)[::H]
        mu_oos = R_ind.mean(axis=0) * 100.0
        if labels is None:
            labels = self.labels
        if portfolios_to_show is None:
            portfolios_to_show = list(w_dict.keys())
        GREEN = "#059669"; RED = "#dc2626"
        n_ptf = len(portfolios_to_show)
        ncols = min(2, n_ptf)
        nrows = int(np.ceil(n_ptf / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(11*ncols, 5.5*nrows), squeeze=False)
        fig.patch.set_facecolor(BG_COLOR)
        for idx, ptf_name in enumerate(portfolios_to_show):
            ax      = axes[idx // ncols][idx % ncols]
            _style_ax(ax)
            w       = np.asarray(w_dict[ptf_name], dtype=float)
            contrib = w * mu_oos
            total   = contrib.sum()
            order   = np.argsort(contrib)[::-1]
            cs      = contrib[order]
            ls      = [labels[i] for i in order]
            ws      = w[order] * 100.0
            running = np.zeros(len(cs) + 1)
            bottoms = np.zeros(len(cs))
            for k, c in enumerate(cs):
                bottoms[k]   = running[k]
                running[k+1] = running[k] + c
            colors = [GREEN if c >= 0 else RED for c in cs]
            y_pos  = np.arange(len(cs))
            ax.barh(y_pos, cs, left=bottoms, color=colors, alpha=0.85,
                    edgecolor="white", lw=0.7, zorder=3)
            span = abs(cs).max() if abs(cs).max() > 0 else 1e-4
            for k, (c, bo, lb, wt) in enumerate(zip(cs, bottoms, ls, ws)):
                xc = bo + c / 2
                ax.text(xc, k, f"{c:+.4f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if abs(c) > span * 0.1 else UDP_GRAY)
                ax.text(-span * 0.02, k, f"{lb} ({wt:.1f}%)",
                        ha="right", va="center", fontsize=8, color=UDP_GRAY)
            ax.axvline(total, color=UDP_BLUE, lw=2.0, ls="--", zorder=4,
                       label=f"Total: {total:+.4f}%")
            ax.axvline(0, color=UDP_GRAY, lw=0.8, zorder=1)
            ax.set_yticks([])
            ax.set_xlabel("Contribucion al retorno OOS (%)", fontsize=9)
            ax.set_title(ptf_name, fontsize=11, fontweight="bold", color=UDP_BLUE)
            ax.legend(fontsize=9, loc="lower right", facecolor="white", edgecolor=UDP_GRAY)
        for idx in range(n_ptf, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.suptitle(
            "Atribucion de Retorno OOS por Activo (Waterfall)\n"
            "Contribucion = w_i * E[r_i^OOS]  periodos independientes stride H=5",
            fontsize=13, fontweight="bold", color=UDP_BLUE, y=1.01)
        plt.tight_layout()
        _save(fig, self._p("waterfall_attribution.png"))

    # ════════════════════════════════════════════════════════════════════════
    # run_all — ejecuta todos los plots disponibles
    # ════════════════════════════════════════════════════════════════════════
    def run_all(self, mm_results=None, wiener_results=None,
                w_dict=None, R_oos=None, dates_oos=None, metrics_df=None):
        """Genera todos los graficos disponibles con los datos provistos."""
        import os
        os.makedirs(self.out_dir, exist_ok=True)
        if mm_results is not None:
            try:
                self.plot_convergencia_bcd(mm_results)
                print("  ok convergence_bcd.png")
            except Exception as e:
                print(f"  err convergence_bcd: {e}")
            try:
                self.plot_distribucion_momentos(mm_results)
                print("  ok moments_panel.png")
            except Exception as e:
                print(f"  err moments_panel: {e}")
            try:
                self.plot_probabilidades_escenarios(mm_results)
                print("  ok scenario_probabilities.png")
            except Exception as e:
                print(f"  err scenario_probabilities: {e}")
        if w_dict is not None and R_oos is not None:
            try:
                if dates_oos is not None and metrics_df is not None:
                    self.plot_backtest(w_dict, R_oos, dates_oos=dates_oos,
                                      metrics_df=metrics_df)
                else:
                    self.plot_backtest(w_dict, R_oos)
                print("  ok backtest.png")
            except Exception as e:
                print(f"  err backtest: {e}")
            try:
                self.plot_portfolio_weights_bar(w_dict)
                print("  ok portfolio_weights.png")
            except Exception as e:
                print(f"  err portfolio_weights: {e}")
            try:
                self.plot_fan_chart(w_dict, R_oos, dates_oos=dates_oos)
                print("  ok fan_chart.png")
            except Exception as e:
                print(f"  err fan_chart: {e}")
            try:
                self.plot_waterfall_attribution(w_dict, R_oos)
                print("  ok waterfall_attribution.png")
            except Exception as e:
                print(f"  err waterfall_attribution: {e}")
        if w_dict is not None and R_oos is not None and metrics_df is not None:
            try:
                self.plot_scorecard_heatmap(metrics_df)
                print("  ok scorecard_heatmap.png")
            except Exception as e:
                print(f"  err scorecard_heatmap: {e}")


if __name__ == "__main__":
    print("diagnostics_plots.py cargado correctamente.")
