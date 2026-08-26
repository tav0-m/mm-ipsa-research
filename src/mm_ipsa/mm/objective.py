"""Objetivo Matching-Moment vectorizado y sus gradientes analiticos.

Ajusta momentos marginales y covarianzas por minimos cuadrados ponderados. No es
una implementacion literal de Ponomareva, Roman y Date (2015) ni de la
reformulacion de Contreras, Bosch y Herrera (2018).
"""

import numpy as np

MOMENT_ORDERS = 4


class MMObjective:
    """Funcion objetivo ``F(x, p)`` con gradientes cerrados.

    ``F`` suma los errores cuadraticos de los cuatro primeros momentos centrales
    y un termino de dependencia sobre las covarianzas cruzadas::

        F(x, p) = sum_k sum_i W[k, i] (momento[k, i] - objetivo[k, i])^2
                + peso_dependencia * sum_{i != j} A[i, j] (C[i, j] - Sigma[i, j])^2

    Parameters
    ----------
    moment_targets : (4, n) momentos centrales historicos.
    covariance_target : (n, n) covarianza historica objetivo.
    weights : pesos por momento y opciones de escalado.
    n_scenarios : tamano del soporte con que se calibra.
    """

    def __init__(
        self,
        moment_targets: np.ndarray,
        covariance_target: np.ndarray,
        weights: dict,
        n_scenarios: int,
    ):
        moment_targets = np.asarray(moment_targets, dtype=float)
        covariance_target = np.asarray(covariance_target, dtype=float)
        if moment_targets.ndim != 2 or moment_targets.shape[0] != MOMENT_ORDERS:
            raise ValueError(f"moment_targets debe tener shape ({MOMENT_ORDERS}, n)")
        n_assets = moment_targets.shape[1]
        if covariance_target.shape != (n_assets, n_assets):
            raise ValueError("covariance_target debe tener shape (n, n)")
        if not np.all(np.isfinite(moment_targets)) or not np.all(
            np.isfinite(covariance_target)
        ):
            raise ValueError("Los targets contienen NaN o infinitos")
        if n_scenarios <= 1:
            raise ValueError("n_scenarios debe ser mayor que 1")

        self.M = moment_targets.copy()
        self.Sigma_tgt = covariance_target.copy()
        self.N = n_scenarios
        self.n = n_assets

        self.moment_weights = np.array(
            [weights["k1"], weights["k2"], weights["k3"], weights["k4"]]
        )
        self.dependence_weight = float(weights.get("cov_weight", 1.0))
        self.CW = self._covariance_scaling(weights)
        self.moment_scales = self._moment_scales(weights)
        self.W = self.moment_weights[:, None] / self.moment_scales**2

    def _covariance_scaling(self, weights: dict) -> np.ndarray:
        """Pesos del termino de dependencia, normalizados por volatilidad.

        Sin normalizar, las covarianzas son de orden 1e-4 y el termino de
        Frobenius se vuelve numericamente irrelevante frente a los momentos
        marginales, que si estan normalizados. La diagonal se excluye porque la
        varianza ya se calibra en el segundo momento, y cada par simetrico
        recibe medio peso para contribuir una sola vez.
        """
        floor = float(weights.get("cov_scale_floor", 1e-3))
        if bool(weights.get("cov_normalize", True)):
            deviations = np.sqrt(np.maximum(np.diag(self.Sigma_tgt), 1e-12))
            scale = np.outer(deviations, deviations)
            scaling = 1.0 / np.maximum(scale, floor) ** 2
        else:
            scaling = np.ones_like(self.Sigma_tgt)

        if bool(weights.get("cov_offdiag_only", True)):
            np.fill_diagonal(scaling, 0.0)
        scaling[~np.eye(self.n, dtype=bool)] *= 0.5
        return scaling

    def _moment_scales(self, weights: dict) -> np.ndarray:
        """Escalas por momento, con piso para evitar pesos explosivos.

        El modo por potencia de volatilidad impide que una media o un tercer
        momento cercanos a cero dominen el objetivo.
        """
        mode = str(weights.get("moment_scale_mode", "volatility_power"))
        floor = float(weights.get("moment_scale_floor", 1e-6))
        if mode == "volatility_power":
            variance = np.maximum(self.M[1], floor**2)
            deviation = np.sqrt(variance)
            scales = np.vstack([deviation, variance, deviation**3, variance**2])
        elif mode == "target_magnitude":
            scales = np.maximum(np.abs(self.M), floor)
        else:
            raise ValueError(
                "moment_scale_mode debe ser 'volatility_power' o 'target_magnitude'"
            )
        return np.maximum(scales, floor)

    def compute_moments(self, x: np.ndarray, p: np.ndarray) -> tuple:
        """Cuatro momentos centrales ponderados, media y covarianza."""
        moments, mean, covariance, _ = self._moments_with_deviations(x, p)
        return moments, mean, covariance

    def _moments_with_deviations(
        self, x: np.ndarray, p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Momentos, media, covarianza y las desviaciones ya calculadas.

        Las potencias se encadenan por multiplicacion en vez de usar ``**``:
        ``np.power`` es un orden de magnitud mas lento, y esta rutina se evalua
        miles de veces por cada busqueda por retroceso del paso-p. La diferencia
        es solo de reasociacion en punto flotante.

        Devolver las desviaciones evita que los gradientes las reconstruyan.
        """
        mean = p @ x
        deviations = x - mean[np.newaxis, :]
        squared = deviations * deviations
        cubed = squared * deviations

        moments = np.empty((MOMENT_ORDERS, self.n))
        moments[0] = mean
        moments[1] = p @ squared
        moments[2] = p @ cubed
        moments[3] = p @ (squared * squared)

        covariance = (p[:, None] * deviations).T @ deviations
        return moments, mean, covariance, deviations

    def evaluate(self, x: np.ndarray, p: np.ndarray) -> float:
        """Valor de ``F(x, p)``."""
        self._validate_state(x, p)
        moments, _, covariance = self.compute_moments(x, p)
        moment_error = moments - self.M
        dependence_error = covariance - self.Sigma_tgt
        return float(
            np.sum(self.W * moment_error**2)
            + self.dependence_weight * np.sum(self.CW * dependence_error**2)
        )

    def components(self, x: np.ndarray, p: np.ndarray) -> dict:
        """Descompone ``F`` por termino, para auditoria entre corridas."""
        self._validate_state(x, p)
        moments, _, covariance = self.compute_moments(x, p)
        moment_error = moments - self.M
        dependence_error = covariance - self.Sigma_tgt
        by_moment = np.sum(self.W * moment_error**2, axis=1)
        dependence = self.dependence_weight * float(
            np.sum(self.CW * dependence_error**2)
        )
        return {
            "mean": float(by_moment[0]),
            "variance": float(by_moment[1]),
            "third_central": float(by_moment[2]),
            "fourth_central": float(by_moment[3]),
            "dependence": dependence,
            "total": float(by_moment.sum() + dependence),
        }

    def _validate_state(self, x: np.ndarray, p: np.ndarray) -> None:
        x = np.asarray(x)
        p = np.asarray(p)
        # N se guarda como el tamano de calibracion, pero la matematica no lo
        # usa: exigirlo impediria evaluar una solucion de ensemble, cuyo soporte
        # es un multiplo de N.
        if x.ndim != 2 or x.shape[1] != self.n:
            raise ValueError(f"x debe tener {self.n} columnas; recibido {x.shape}")
        if p.ndim != 1 or p.shape[0] != x.shape[0]:
            raise ValueError("p debe tener un valor por escenario de x")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
            raise ValueError("x o p contienen NaN o infinitos")
        if np.min(p) < -1e-10:
            raise ValueError("p contiene probabilidades negativas")
        # Los solvers con restricciones evaluan puntos intermedios apenas fuera
        # del hiperplano; la salida final se valida con tolerancia mas estricta.
        if abs(float(p.sum()) - 1.0) > 1e-4:
            raise ValueError("p debe sumar aproximadamente 1 durante la optimizacion")

    def grad_x(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Gradiente respecto de las posiciones, con shape ``(N, n)``.

        Derivacion analitica del objetivo definido en este modulo::

            dF/dx[s, r] = p[s] * (
                2 W[0, r] e[0, r]
              + 4 W[1, r] e[1, r] dev[s, r]
              + 6 W[2, r] e[2, r] (dev[s, r]^2 - m[1, r])
              + 8 W[3, r] e[3, r] (dev[s, r]^3 - m[2, r])
              + 4 w_dep sum_l A[r, l] dC[r, l] dev[s, l]
            )
        """
        moments, _, covariance, deviations = self._moments_with_deviations(x, p)
        squared = deviations * deviations
        moment_error = moments - self.M
        dependence_error = self.CW * (covariance - self.Sigma_tgt)
        probability = p[:, None]

        gradient = 2 * self.W[0] * moment_error[0] * probability
        gradient += 4 * self.W[1] * moment_error[1] * probability * deviations
        gradient += (
            6 * self.W[2] * moment_error[2] * probability * (squared - moments[1])
        )
        gradient += (
            8
            * self.W[3]
            * moment_error[3]
            * probability
            * (squared * deviations - moments[2])
        )
        # El Frobenius simetrico aporta dos terminos equivalentes, de ahi el 4.
        gradient += (
            4 * self.dependence_weight * probability * deviations @ dependence_error.T
        )
        return gradient

    def grad_p(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Gradiente respecto de las probabilidades, con shape ``(N,)``::

            dF/dp[s] = sum_i (
                2 W[0, i] e[0, i] x[s, i]
              + 2 W[1, i] e[1, i] dev[s, i]^2
              + 2 W[2, i] e[2, i] (dev[s, i]^3 - 3 x[s, i] m[1, i])
              + 2 W[3, i] e[3, i] (dev[s, i]^4 - 4 x[s, i] m[2, i])
            ) + 2 w_dep sum_ij A[i, j] dC[i, j] dev[s, i] dev[s, j]
        """
        moments, _, covariance, deviations = self._moments_with_deviations(x, p)
        squared = deviations * deviations
        cubed = squared * deviations
        moment_error = moments - self.M
        dependence_error = self.CW * (covariance - self.Sigma_tgt)

        gradient = 2 * x @ (self.W[0] * moment_error[0])
        gradient += 2 * squared @ (self.W[1] * moment_error[1])
        gradient += 2 * (cubed - 3 * x * moments[1]) @ (self.W[2] * moment_error[2])
        gradient += 2 * (squared * squared - 4 * x * moments[2]) @ (
            self.W[3] * moment_error[3]
        )
        gradient += 2 * self.dependence_weight * np.einsum(
            "si,ij,sj->s", deviations, dependence_error, deviations
        )
        return gradient
