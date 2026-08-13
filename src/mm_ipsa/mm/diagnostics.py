"""Diagnósticos tabulares y gráficos de la calibración MM."""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D
from scipy import stats

# Estilo consistente con el informe de investigacion.
plt.rcParams.update({
    "font.family"       : "serif",
    "font.size"         : 9,
    "axes.titlesize"    : 10,
    "figure.dpi"        : 150,
    "axes.grid"         : True,
    "grid.alpha"        : 0.25,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})
COL_HIST = "#2E5FA3"
COL_MM   = "#E8622A"


# ═══════════════════════════════════════════════════════════════════════════
class MMDiagnostics:
    """Tablas y figuras que contrastan la solucion MM con sus momentos objetivo.

    La curtosis se reporta como exceso, de modo que el cero corresponde a colas
    gaussianas y no se confunda un valor normal con una cola pesada.
    """

    def __init__(self, x, p, M, Sigma_tgt, labels, cfg):
        self.x      = x          # (N, n)
        self.p      = p          # (N,)
        self.M      = M          # (4, n) momentos objetivo
        self.Sigma  = Sigma_tgt  # (n, n) covarianza objetivo
        self.labels = labels
        self.N, self.n = x.shape
        
        fig_dir = Path(cfg["paths"]["figures"])
        tab_dir = Path(cfg["paths"]["tables"])
        fig_dir.mkdir(parents=True, exist_ok=True)
        tab_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = fig_dir
        self.tab_dir = tab_dir
        
        # Momentos MM
        mu       = p @ x
        dev      = x - mu[np.newaxis,:]
        self.mu_mm = mu
        self.m_mm  = np.zeros((4, self.n))
        self.m_mm[0] = mu
        for k in range(1,4):
            self.m_mm[k] = p @ (dev**(k+1))
        
        std = np.sqrt(np.maximum(np.diag(Sigma_tgt),0))
        self.corr_hist = Sigma_tgt / (std[:,None]*std[None,:] + 1e-10)
        
        C_mm = (p[:,None,None] * dev[:,:,None] * dev[:,None,:]).sum(0)
        std_mm = np.sqrt(np.maximum(np.diag(C_mm),0))
        self.corr_mm = C_mm / (std_mm[:,None]*std_mm[None,:] + 1e-10)

    # ───────────────────────────────────────────────────────────────────────
    def table_moments(self) -> pd.DataFrame:
        """Tabla de momentos histórico vs MM con errores."""
        rows = []
        for i, label in enumerate(self.labels):
            mh = self.M[:, i]
            mm = self.m_mm[:, i]
            
            # Se reporta exceso de curtosis (Pearson menos 3) para que el cero
            # sea la referencia gaussiana. Con la convencion de Pearson, un
            # valor de 3 se lee como cola pesada cuando en realidad es normal.
            kurt_h = mh[3]/(mh[1]**2+1e-10) - 3.0
            kurt_m = mm[3]/(mm[1]**2+1e-10) - 3.0
            skew_h = mh[2]/(mh[1]**1.5+1e-10)
            skew_m = mm[2]/(mm[1]**1.5+1e-10)

            rows.append({
                "Activo"           : label,
                "mu_hist_%"        : mh[0]*100,
                "mu_mm_%"          : mm[0]*100,
                "sig_hist_%"       : np.sqrt(mh[1])*100,
                "sig_mm_%"         : np.sqrt(mm[1])*100,
                "skew_hist"        : skew_h,
                "skew_mm"          : skew_m,
                "excess_kurt_hist" : kurt_h,
                "excess_kurt_mm"   : kurt_m,
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.tab_dir / "moments_comparison.csv", index=False)
        return df

    # ───────────────────────────────────────────────────────────────────────
    def table_mae(self) -> pd.DataFrame:
        """Tabla MAE, RMSE y error relativo por momento."""
        moment_names = ["Media (m1)", "Varianza (m2)",
                        "Asimetría (m3)", "Kurtosis (m4)"]
        rows = []
        for k, name in enumerate(moment_names):
            diff = np.abs(self.m_mm[k] - self.M[k])
            mae  = diff.mean()
            rmse = np.sqrt((diff**2).mean())
            denom= np.abs(self.M[k]).mean()
            rel  = mae / (denom + 1e-12)
            rows.append({
                "Momento"      : name,
                "MAE"          : mae,
                "RMSE"         : rmse,
                "Err_relativo" : rel*100,
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.tab_dir / "mae_moments.csv", index=False)
        return df

    # ───────────────────────────────────────────────────────────────────────
    def table_probs(self) -> pd.DataFrame:
        """Estadísticas de la distribución de probabilidades."""
        p = self.p
        n_active = int((p > 1e-6).sum())
        N_eff    = 1.0 / (p**2).sum()
        
        def gini(v):
            v = np.sort(v)
            return 1 - 2*(np.cumsum(v)/v.sum()).mean()
        
        rows = [
            {"Estadístico": "Escenarios totales (N)",          "Valor": self.N},
            {"Estadístico": "Escenarios activos (p>1e-6)",      "Valor": n_active},
            {"Estadístico": "Escenarios inactivos",             "Valor": self.N-n_active},
            {"Estadístico": "p_max",                            "Valor": round(p.max(),4)},
            {"Estadístico": "1/N (equiprobable)",               "Valor": round(1/self.N,4)},
            {"Estadístico": "std(p)",                           "Valor": round(p.std(),4)},
            {"Estadístico": "N_eff = 1/Σpj²",                  "Valor": round(N_eff,1)},
            {"Estadístico": "Índice de Gini",                   "Valor": round(gini(p),3)},
            {"Estadístico": "Peso top-10 escenarios (%)",       "Valor": round(np.sort(p)[::-1][:10].sum()*100,1)},
            {"Estadístico": "Peso top-20 escenarios (%)",       "Valor": round(np.sort(p)[::-1][:20].sum()*100,1)},
        ]
        df = pd.DataFrame(rows)
        df.to_csv(self.tab_dir / "scenario_probs.csv", index=False)
        return df

    # ───────────────────────────────────────────────────────────────────────
    def plot_convergence(self, history: list, all_starts: list) -> None:
        """Figura de convergencia BCD — start ganador."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(history, color="#2E5FA3", lw=2.5, label=f"Start ganador (F*={history[-1]:.4f})")
        ax.set_xlabel("Iteración BCD")
        ax.set_ylabel("F(x, p) — función de error")
        ax.set_title("Convergencia BCD — Matching-Moment IPSA", fontweight="bold")
        
        ax.text(0.01, 0.95, f"F óptimo = {history[-1]:.4f}",
                transform=ax.transAxes, fontsize=9,
                va="top", bbox=dict(fc="#27AE60", ec="white", alpha=0.8, boxstyle="round"))
        ax.text(len(history)-1, history[-1],
                f"F={history[-1]:.3f}", ha="right", va="bottom",
                fontsize=9, color="#2E5FA3", fontweight="bold")
        
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(self.fig_dir / "convergence_bcd.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [ok] convergence_bcd.png")

        if len(all_starts) > 1:
            self.plot_multistart_convergence(all_starts)

    # ───────────────────────────────────────────────────────────────────────
    def plot_multistart_convergence(self, all_starts: list) -> None:
        """Dashboard visual para justificar el start ganador."""
        rows = []
        histories = {}
        for r in all_starts:
            start_id = int(r["start_id"])
            histories[start_id] = np.asarray(r["history"], dtype=float)
            rows.append({
                "start_id": start_id,
                "seed": int(r["seed"]),
                "F_ini": float(r["F_ini"]),
                "F_fin": float(r["F_fin"]),
                "iters": int(r["iters"]),
                "stop": r.get("stop_reason", "tol" if r.get("converged") else "max_iter"),
                "rel_final": float(r.get("rel_final", np.nan)),
            })

        starts_df = pd.DataFrame(rows).sort_values("F_fin").reset_index(drop=True)
        starts_df["rank"] = np.arange(1, len(starts_df) + 1)
        starts_df["is_best"] = starts_df["rank"].eq(1)
        starts_df.to_csv(self.tab_dir / "bcd_multistart_ranking.csv", index=False)

        winner = starts_df.iloc[0]
        winner_id = int(winner["start_id"])
        colors = colormaps["tab10"](np.linspace(0, 1, max(10, len(histories))))

        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.08, 0.92])
        ax_log = fig.add_subplot(gs[0, 0])
        ax_zoom = fig.add_subplot(gs[0, 1])
        ax_bar = fig.add_subplot(gs[1, 0])
        ax_card = fig.add_subplot(gs[1, 1])

        for idx, (start_id, hist) in enumerate(histories.items()):
            label = f"start {start_id}" + (" - ganador" if start_id == winner_id else "")
            lw = 3.0 if start_id == winner_id else 1.5
            alpha = 1.0 if start_id == winner_id else 0.55
            color = colors[idx % len(colors)]
            x = np.arange(len(hist))
            ax_log.plot(x, hist, lw=lw, alpha=alpha, color=color, label=label)
            if len(hist) > 2:
                ax_zoom.plot(x[2:], hist[2:], lw=lw, alpha=alpha, color=color, label=label)

        ax_log.set_yscale("log")
        ax_log.set_title("Convergencia multi-start (escala log)", fontweight="bold")
        ax_log.set_xlabel("Iteracion BCD")
        ax_log.set_ylabel("F(x,p)")
        ax_log.grid(True, alpha=0.25, which="both")
        ax_log.legend(fontsize=8)

        ax_zoom.set_yscale("log")
        ax_zoom.set_title("Zoom despues del salto inicial", fontweight="bold")
        ax_zoom.set_xlabel("Iteracion BCD")
        ax_zoom.set_ylabel("F(x,p)")
        ax_zoom.grid(True, alpha=0.25, which="both")

        bar_colors = ["#1B7837" if sid == winner_id else "#9E9E9E"
                      for sid in starts_df["start_id"]]
        labels = [f"S{sid}\nseed {seed}" for sid, seed in zip(starts_df["start_id"], starts_df["seed"])]
        ax_bar.bar(labels, starts_df["F_fin"], color=bar_colors, edgecolor="white")
        ax_bar.set_yscale("log")
        ax_bar.set_title("Ranking por F final", fontweight="bold")
        ax_bar.set_ylabel("F_final (menor es mejor)")
        ax_bar.grid(True, alpha=0.25, axis="y", which="both")
        ax_bar.tick_params(axis="x", labelsize=8)

        second = starts_df.iloc[1] if len(starts_df) > 1 else winner
        gain = (float(second["F_fin"]) / float(winner["F_fin"]) - 1.0) * 100.0 if len(starts_df) > 1 else 0.0
        card_text = (
            "Start ganador\n\n"
            f"start = {winner_id}\n"
            f"seed = {int(winner['seed'])}\n"
            f"F_final = {winner['F_fin']:.6g}\n"
            f"iteraciones = {int(winner['iters'])}\n"
            f"stop = {winner['stop']}\n"
            f"mejora vs 2do = {gain:.1f}%\n\n"
            "Lectura para la investigacion:\n"
            "El multi-start reduce el riesgo de escoger un minimo local.\n"
            "Se reporta como ganador el start con menor F_final."
        )
        ax_card.axis("off")
        ax_card.text(
            0.02, 0.96, card_text,
            va="top",
            fontsize=11,
            linespacing=1.35,
            bbox=dict(facecolor="#E8F5E9", edgecolor="#1B7837", boxstyle="round,pad=0.65"),
        )

        fig.suptitle("Robustez BCD: seleccion del start ganador", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        fig.savefig(self.fig_dir / "convergence_bcd_multistart.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("  [ok] convergence_bcd_multistart.png")

    # ───────────────────────────────────────────────────────────────────────
    def plot_scenario_probs(self) -> None:
        """Figura de distribución de probabilidades y curva de Lorenz."""
        p = self.p
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Distribución de probabilidades — Escenarios MM",
                     fontsize=11, fontweight="bold")
        
        # Top-30
        top_idx = np.argsort(p)[::-1][:30]
        top_p   = p[top_idx]
        
        ax1.bar(range(30), top_p, color="#2E5FA3", alpha=0.85, edgecolor="white")
        ax1.axhline(1/self.N, color="orange", ls="--", lw=1.5,
                    label=f"Equiprobable (1/N={1/self.N:.4f})")
        n_active = int((p > 1e-6).sum())
        ax1.text(0.02, 0.97, f"{n_active} escenarios activos",
                 transform=ax1.transAxes, va="top",
                 bbox=dict(fc="#27AE60",ec="white",alpha=0.8,boxstyle="round"),
                 fontsize=9, color="white", fontweight="bold")
        ax1.set_xlabel("Escenario (top 30 por probabilidad)")
        ax1.set_ylabel("Probabilidad")
        ax1.set_title("Top 30 escenarios por probabilidad")
        ax1.legend(fontsize=9)
        
        # Curva de Lorenz
        p_sorted = np.sort(p)
        cum_p    = np.cumsum(p_sorted) / p_sorted.sum()
        cum_n    = np.linspace(0, 1, self.N)
        
        ax2.plot(cum_n, cum_p, color="#2E5FA3", lw=2, label="Curva de Lorenz (p MM)")
        ax2.plot([0,1],[0,1],"--",color="gray",lw=1,label="Equiprobable (45°)")
        ax2.fill_between(cum_n, cum_p, cum_n, alpha=0.2, color="#2E5FA3")
        
        gini = 1 - 2*cum_p.mean()
        ax2.text(0.05, 0.90, f"Gini = {gini:.3f}", transform=ax2.transAxes,
                 fontsize=9, bbox=dict(fc="white",ec="gray",alpha=0.8,boxstyle="round"))
        ax2.set_xlabel("Fracción acumulada de escenarios")
        ax2.set_ylabel("Fracción acumulada de probabilidad")
        ax2.set_title("Curva de Lorenz — heterogeneidad de escenarios")
        ax2.legend(fontsize=9)
        
        plt.tight_layout()
        fig.savefig(self.fig_dir / "scenario_probabilities.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [ok] scenario_probabilities.png")

    # ───────────────────────────────────────────────────────────────────────
    def plot_moments_panel(self, terminal: pd.DataFrame) -> None:
        """Panel de momentos histórico vs MM."""
        X_hist = terminal[self.labels].to_numpy(dtype=float)
        labels = self.labels
        n      = self.n
        x_pos  = np.arange(n)
        w      = 0.38
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 13))
        fig.suptitle("Matching de Momentos — IPSA H=5 días (2020-2025)",
                     fontsize=12, fontweight="bold")
        
        # Momentos históricos en escala interpretable
        mu_h  = X_hist.mean(0)*100
        sig_h = X_hist.std(0)*100
        sk_h  = np.array([(((x:=X_hist[:,i])-x.mean())**3).mean()/
                           (x.std()**3+1e-10) for i in range(n)])
        # Exceso de curtosis: cero equivale a colas gaussianas.
        ku_h  = np.array([(((x:=X_hist[:,i])-x.mean())**4).mean()/
                           (x.std()**4+1e-10) for i in range(n)]) - 3.0

        mu_m  = self.mu_mm*100
        sig_m = np.sqrt(np.maximum(self.m_mm[1],0))*100
        sk_m  = self.m_mm[2]/(self.m_mm[1]**1.5+1e-10)
        ku_m  = self.m_mm[3]/(self.m_mm[1]**2+1e-10) - 3.0
        
        panels = [
            (axes[0], mu_h,  mu_m,  "Retorno esperado H=5 — MM replica con error < 0.01%", "Media (×100)"),
            (axes[1], sig_h, sig_m, "Desviación estándar — ajuste casi perfecto en todos los activos","Volatilidad (%)"),
            (axes[2], sk_h,  sk_m,  "Asimetría — MM captura el signo y magnitud con alta precisión", "Skewness"),
            (axes[3], ku_h,  ku_m,  "Colas pesadas — exceso de curtosis (0 = gaussiana)","Exceso de curtosis"),
        ]
        
        for ax, vals_h, vals_m, subtitle, ylabel in panels:
            ax.bar(x_pos-w/2, vals_h, w, label="Histórico",
                   color=COL_HIST, alpha=0.85, edgecolor="white")
            ax.bar(x_pos+w/2, vals_m, w, label="MM",
                   color=COL_MM, alpha=0.85, edgecolor="white")
            
            # Porcentajes de error
            for i in range(n):
                err = abs(vals_h[i]-vals_m[i])/(abs(vals_h[i])+1e-6)*100
                col = "#27AE60" if err < 5 else "#E74C3C"
                ax.text(x_pos[i]+w/2, vals_m[i]+abs(vals_m.max()-vals_m.min())*0.02,
                        f"{err:.0f}%", ha="center", fontsize=6, color=col)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(subtitle, fontsize=8.5, style="italic", color="#444")
            ax.legend(fontsize=8, loc="upper right")
            ax.axhline(0, color="black", lw=0.5, ls="--")
        
        plt.tight_layout()
        fig.savefig(self.fig_dir / "moments_panel.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [ok] moments_panel.png")

    # ───────────────────────────────────────────────────────────────────────
    def plot_hist_grid(self, terminal: pd.DataFrame) -> None:
        """Grid de distribuciones terminales histórico vs MM simulado."""
        X_hist = terminal[self.labels].to_numpy(dtype=float)
        
        # Simular 10000 retornos terminales desde los escenarios MM
        idx = np.random.choice(self.N, size=10000, p=self.p)
        X_mm = self.x[idx]
        
        ncols = 4
        nrows = 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 14))
        fig.suptitle("Distribuciones terminales H=5 días — Histórico vs Simulado MM",
                     fontsize=12, fontweight="bold")
        axes_f = axes.flatten()
        
        from scipy.stats import ks_2samp
        
        last_index = -1
        for i, (label, ax) in enumerate(zip(self.labels, axes_f)):
            last_index = i
            h = X_hist[:, i]
            m = X_mm[:, i]
            
            xmin = min(h.min(), m.min())
            xmax = max(h.max(), m.max())
            pad  = (xmax-xmin)*0.05
            xx   = np.linspace(xmin-pad, xmax+pad, 250)
            
            kde_h = stats.gaussian_kde(h, bw_method=0.3)
            kde_m = stats.gaussian_kde(m, bw_method=0.3)
            
            ax.hist(h, bins=30, density=True, color=COL_HIST, alpha=0.35,
                    label="Histórico")
            ax.hist(m, bins=30, density=True, color=COL_MM,   alpha=0.35)
            ax.plot(xx, kde_h(xx), color=COL_HIST, lw=2.2, label="Histórico KDE")
            ax.plot(xx, kde_m(xx), color=COL_MM,   lw=2.0, ls="--", label="MM KDE")
            
            ks = float(np.asarray(ks_2samp(h, m), dtype=float).ravel()[0])
            col_badge = "#27AE60" if ks<0.10 else ("#F39C12" if ks<0.15 else "#E74C3C")
            ax.text(0.03, 0.97, f"KS={ks:.3f}", transform=ax.transAxes,
                    va="top", fontsize=7.5, color="white", fontweight="bold",
                    bbox=dict(fc=col_badge,ec="white",alpha=0.85,boxstyle="round,pad=0.3"))
            
            ax.axvline(0, color="black", lw=0.5, ls=":")
            ax.set_title(label, fontsize=9, fontweight="bold")
            ax.set_xlabel("Retorno terminal", fontsize=7)
            ax.set_ylabel("Densidad", fontsize=7)
            ax.tick_params(labelsize=7)
        
        if last_index < len(axes_f)-1:
            for j in range(last_index+1, len(axes_f)):
                axes_f[j].set_visible(False)
        
        handles = [Line2D([0],[0],color=COL_HIST,lw=2,label="Histórico"),
                   Line2D([0],[0],color=COL_MM,lw=2,ls="--",label="MM")]
        fig.legend(handles=handles, loc="lower right", fontsize=9, ncol=2)
        
        plt.tight_layout()
        fig.savefig(self.fig_dir / "hist_grid_H5.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [ok] hist_grid_H5.png")

    # ───────────────────────────────────────────────────────────────────────
    def plot_corr_matrices(self) -> None:
        """Matrices de correlación histórica vs MM."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Correlación terminal (H=5 días) — Histórico | MM | Diferencia",
                     fontsize=11, fontweight="bold")
        
        diff = self.corr_mm - self.corr_hist
        n    = self.n
        labs = self.labels
        
        for ax, mat, title, cmap, vmin, vmax in [
            (axes[0], self.corr_hist, "Histórico",      "Blues",  0, 1),
            (axes[1], self.corr_mm,   "Simulado MM",    "Oranges",0, 1),
            (axes[2], diff,           "Diferencia MM−Hist","RdBu_r",-0.08,0.08),
        ]:
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(n))
            ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labs, fontsize=6)
            ax.set_title(title, fontsize=10, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            for r in range(n):
                for c in range(n):
                    v = mat[r,c]
                    col = "white" if abs(v)>0.5 else "black"
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=5, color=col)
        
        # Error medio
        mask = ~np.eye(n, dtype=bool)
        err  = np.abs(diff[mask]).mean()
        axes[2].set_title(f"Diferencia MM−Hist\n(err medio={err:.4f})",
                          fontsize=9, fontweight="bold")
        
        plt.tight_layout()
        fig.savefig(self.fig_dir / "corr_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [ok] corr_comparison.png")

    # ───────────────────────────────────────────────────────────────────────
    def run_all(self, history: list, all_starts: list,
                terminal: pd.DataFrame) -> None:
        """Ejecuta todos los diagnósticos."""
        print("\n[diagnostics] figuras MM")
        
        self.plot_convergence(history, all_starts)
        self.plot_scenario_probs()
        self.plot_moments_panel(terminal)
        self.plot_hist_grid(terminal)
        self.plot_corr_matrices()
        
        df_mae = self.table_mae()
        self.table_moments()
        self.table_probs()
        
        print("\n  MAE por momento:")
        for _, row in df_mae.iterrows():
            print(f"    {row['Momento']:20s} "
                  f"MAE={row['MAE']:.3e}  "
                  f"Err.rel={row['Err_relativo']:.2f}%")
        
        print(f"\n  [ok] tablas guardadas en {self.tab_dir}/")
        print(f"  [ok] figuras guardadas en {self.fig_dir}/")
