"""Objetivo Matching-Moment vectorizado y sus gradientes analíticos.

Esta extensión ajusta momentos marginales y covarianzas mediante mínimos
cuadrados. No es una implementación literal de Ponomareva, Roman y Date (2015)
ni de la reformulación de Contreras, Bosch y Herrera (2018).
"""

import numpy as np


class MMObjective:
    """
    Función objetivo F(x,p) del Matching-Moment con gradientes vectorizados.

    F(x, p) = Σ_{k=1}^{4} w_k · Σ_i (m̂_{k,i} − M_{k,i})²
            + w_Σ · ‖Ĉ − Σ‖²_F

    Parameters
    ----------
    M        : (4, n)  momentos centrales históricos
    Sigma_tgt: (n, n)  covarianza histórica objetivo
    weights  : dict    k1, k2, k3, k4, cov_weight
    N        : int     número de escenarios
    """

    def __init__(self, M: np.ndarray, Sigma_tgt: np.ndarray,
                 weights: dict, N: int):
        M = np.asarray(M, dtype=float)
        Sigma_tgt = np.asarray(Sigma_tgt, dtype=float)
        if M.ndim != 2 or M.shape[0] != 4:
            raise ValueError("M debe tener shape (4, n)")
        if Sigma_tgt.shape != (M.shape[1], M.shape[1]):
            raise ValueError("Sigma_tgt debe tener shape (n, n)")
        if not np.all(np.isfinite(M)) or not np.all(np.isfinite(Sigma_tgt)):
            raise ValueError("Los targets contienen NaN o infinitos")
        if N <= 1:
            raise ValueError("N debe ser mayor que 1")

        self.M         = M.copy()
        self.Sigma_tgt = Sigma_tgt.copy()
        self.w         = np.array([weights["k1"], weights["k2"],
                                   weights["k3"], weights["k4"]])
        self.w_cov     = float(weights.get("cov_weight", 1.0))
        self.cov_normalize = bool(weights.get("cov_normalize", True))
        self.cov_scale_floor = float(weights.get("cov_scale_floor", 1e-3))
        self.cov_offdiag_only = bool(weights.get("cov_offdiag_only", True))
        self.moment_scale_mode = str(
            weights.get("moment_scale_mode", "volatility_power")
        )
        self.moment_scale_floor = float(weights.get("moment_scale_floor", 1e-6))
        self.N         = N
        self.n         = M.shape[1]

        # Scale covariance errors by target volatilities. Without this,
        # covariance entries are O(1e-4) and the Frobenius term can become
        # numerically irrelevant versus normalized marginal moments.
        std_tgt = np.sqrt(np.maximum(np.diag(self.Sigma_tgt), 1e-12))
        cov_scale = np.outer(std_tgt, std_tgt)
        self.CW = 1.0 / np.maximum(cov_scale, self.cov_scale_floor) ** 2
        if not self.cov_normalize:
            self.CW = np.ones_like(self.Sigma_tgt)

        # La varianza ya se calibra mediante M[1]. Se excluye la diagonal del
        # termino de dependencia para no contarla dos veces. Cada par simetrico
        # off-diagonal recibe 0.5 para que contribuya una sola vez.
        if self.cov_offdiag_only:
            np.fill_diagonal(self.CW, 0.0)
        offdiag = ~np.eye(self.n, dtype=bool)
        self.CW[offdiag] *= 0.5

        # Escalas estables por potencia de volatilidad. Evitan que una media o
        # un tercer momento cercanos a cero produzcan pesos explosivos.
        if self.moment_scale_mode == "volatility_power":
            variance = np.maximum(self.M[1], self.moment_scale_floor**2)
            sigma = np.sqrt(variance)
            scales = np.vstack([sigma, variance, sigma**3, variance**2])
        elif self.moment_scale_mode == "target_magnitude":
            scales = np.maximum(np.abs(self.M), self.moment_scale_floor)
        else:
            raise ValueError(
                "moment_scale_mode debe ser 'volatility_power' o "
                "'target_magnitude'"
            )
        self.moment_scales = np.maximum(scales, self.moment_scale_floor)
        self.W = self.w[:, None] / self.moment_scales**2

    # ──────────────────────────────────────────────────────────────────────
    def compute_moments(self, x: np.ndarray,
                        p: np.ndarray) -> tuple:
        """
        Calcula los 4 momentos centrales y la covarianza.

        Parameters
        ----------
        x : (N, n)  posiciones de escenarios
        p : (N,)    probabilidades  (suma = 1, p ≥ 0)

        Returns
        -------
        m  : (4, n)  momentos centrales
        mu : (n,)    media ponderada
        C  : (n, n)  covarianza ponderada
        """
        mu  = p @ x                          # (n,)
        dev = x - mu[np.newaxis, :]          # (N, n)

        m    = np.empty((4, self.n))
        m[0] = mu
        for k in range(1, 4):
            m[k] = p @ (dev ** (k + 1))     # (n,)

        # C_{il} = Σ_j p_j · dev_{ji} · dev_{jl}
        # Vectorizado: C = devᵀ · diag(p) · dev
        C = (p[:, None] * dev).T @ dev      # (n, n)  ← rápido

        return m, mu, C

    # ──────────────────────────────────────────────────────────────────────
    def evaluate(self, x: np.ndarray, p: np.ndarray) -> float:
        """Evalúa F(x, p)."""
        self._validate_state(x, p)
        m, mu, C = self.compute_moments(x, p)
        diff_m = m - self.M
        diff_C = C - self.Sigma_tgt
        return float(np.sum(self.W * diff_m ** 2) +
                     self.w_cov * np.sum(self.CW * diff_C ** 2))

    def components(self, x: np.ndarray, p: np.ndarray) -> dict:
        """Descompone F para auditoría y comparación entre corridas."""
        self._validate_state(x, p)
        m, _, C = self.compute_moments(x, p)
        diff_m = m - self.M
        diff_C = C - self.Sigma_tgt
        moment_terms = np.sum(self.W * diff_m**2, axis=1)
        dependence = self.w_cov * float(np.sum(self.CW * diff_C**2))
        return {
            "mean": float(moment_terms[0]),
            "variance": float(moment_terms[1]),
            "third_central": float(moment_terms[2]),
            "fourth_central": float(moment_terms[3]),
            "dependence": dependence,
            "total": float(moment_terms.sum() + dependence),
        }

    def _validate_state(self, x: np.ndarray, p: np.ndarray) -> None:
        x = np.asarray(x)
        p = np.asarray(p)
        if x.shape != (self.N, self.n):
            raise ValueError(f"x debe tener shape {(self.N, self.n)}")
        if p.shape != (self.N,):
            raise ValueError(f"p debe tener shape {(self.N,)}")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
            raise ValueError("x o p contienen NaN o infinitos")
        if np.min(p) < -1e-10:
            raise ValueError("p contiene probabilidades negativas")
        # Los solvers con restricciones evaluan puntos intermedios apenas fuera
        # del hiperplano. La salida final se valida con tolerancia mas estricta.
        if abs(float(p.sum()) - 1.0) > 1e-4:
            raise ValueError("p debe sumar aproximadamente 1 durante la optimizacion")

    # ──────────────────────────────────────────────────────────────────────
    def grad_x(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Gradiente ∂F/∂x  —  shape (N, n).

        Derivacion analitica de la funcion definida en este modulo,
        completamente vectorizada sin bucles Python.

        ∂F/∂x_{sr} = p_s · [
            2 W₁ᵣ δm₁ᵣ
          + 4 W₂ᵣ δm₂ᵣ · devₛᵣ
          + 6 W₃ᵣ δm₃ᵣ · (devₛᵣ² − m₂ᵣ)
          + 8 W₄ᵣ δm₄ᵣ · (devₛᵣ³ − m₃ᵣ)
          + 4 wΣ · Σₗ δCᵣₗ · devₛₗ
        ]
        """
        m, mu, C = self.compute_moments(x, p)
        dev    = x - mu[np.newaxis, :]      # (N, n)
        diff_m = m - self.M                  # (4, n)
        diff_C = C - self.Sigma_tgt          # (n, n)
        diff_C_w = self.CW * diff_C          # scaled covariance residuals
        ps     = p[:, None]                  # (N, 1)  para broadcasting

        # ── Términos de momentos ─────────────────────────────────────────
        g  = 2 * self.W[0] * diff_m[0] * ps                         # m1
        g += 4 * self.W[1] * diff_m[1] * ps * dev                   # m2
        g += 6 * self.W[2] * diff_m[2] * ps * (dev**2 - m[1])       # m3
        g += 8 * self.W[3] * diff_m[3] * ps * (dev**3 - m[2])       # m4

        # ── Término de covarianza ────────────────────────────────────────
        # El Frobenius simetrico aporta dos terminos equivalentes.
        g += 4 * self.w_cov * ps * dev @ diff_C_w.T                 # cov

        return g                                                      # (N, n)

    # ──────────────────────────────────────────────────────────────────────
    def grad_p(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Gradiente ∂F/∂p  —  shape (N,).

        ∂F/∂p_s = Σᵢ [
            2 W₁ᵢ δm₁ᵢ · xₛᵢ
          + 2 W₂ᵢ δm₂ᵢ · devₛᵢ²
          + 2 W₃ᵢ δm₃ᵢ · (devₛᵢ³ − 3 xₛᵢ m₂ᵢ)
          + 2 W₄ᵢ δm₄ᵢ · (devₛᵢ⁴ − 4 xₛᵢ m₃ᵢ)
        ] + 2wΣ · Σᵢⱼ δCᵢⱼ · devₛᵢ · devₛⱼ
        """
        m, mu, C = self.compute_moments(x, p)
        dev    = x - mu[np.newaxis, :]       # (N, n)
        diff_m = m - self.M                   # (4, n)
        diff_C = C - self.Sigma_tgt           # (n, n)
        diff_C_w = self.CW * diff_C           # scaled covariance residuals

        # ── Términos de momentos  [(N,n) @ (n,)] → (N,) ─────────────────
        g  = 2 * x               @ (self.W[0] * diff_m[0])   # m1
        g += 2 * (dev**2)        @ (self.W[1] * diff_m[1])   # m2
        g += 2 * (dev**3 - 3*x*m[1]) @ (self.W[2] * diff_m[2]) # m3
        g += 2 * (dev**4 - 4*x*m[2]) @ (self.W[3] * diff_m[3]) # m4

        # ── Término de covarianza: einsum 'si,ij,sj->s' ──────────────────
        # diag( dev · diff_C · devᵀ )
        g += 2 * self.w_cov * np.einsum("si,ij,sj->s", dev, diff_C_w, dev)

        return g                                                       # (N,)

    # ──────────────────────────────────────────────────────────────────────
    def compute_errors(self, x: np.ndarray, p: np.ndarray) -> dict:
        """
        Diagnóstico post-calibración: MAE, RMSE y error relativo
        por momento, más error de correlación.
        """
        m, mu, C = self.compute_moments(x, p)
        n = self.n

        errors = {}
        moment_names = ["media", "varianza", "asimetria", "kurtosis"]

        for k, name in enumerate(moment_names):
            diff = np.abs(m[k] - self.M[k])
            mae  = diff.mean()
            denom= np.abs(self.M[k]).mean()
            errors[name] = {
                "MAE"       : mae,
                "RMSE"      : float(np.sqrt((diff**2).mean())),
                "rel_error" : mae / (denom + 1e-12),
                "by_asset"  : diff,
                "hist_vals" : self.M[k].copy(),
                "mm_vals"   : m[k].copy(),
            }

        # Error de correlación off-diagonal
        std_h  = np.sqrt(np.maximum(np.diag(self.Sigma_tgt), 0))
        std_m  = np.sqrt(np.maximum(np.diag(C), 0))
        err_c  = []
        for i in range(n):
            for j in range(i+1, n):
                rho_h = self.Sigma_tgt[i,j] / (std_h[i]*std_h[j] + 1e-10)
                rho_m = C[i,j]              / (std_m[i]*std_m[j] + 1e-10)
                err_c.append(abs(rho_m - rho_h))

        errors["correlacion"] = {
            "MAE" : float(np.mean(err_c)),
            "MAX" : float(np.max(err_c)),
        }
        return errors
