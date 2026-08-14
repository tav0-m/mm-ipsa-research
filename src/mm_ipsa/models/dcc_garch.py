"""Control DCC-GARCH: volatilidad y correlacion condicionales dinamicas.

Los otros tres controles son distribuciones estaticas ajustadas a momentos EWMA:
ninguno modela como evoluciona la volatilidad ni la dependencia dentro del
horizonte. DCC-GARCH (Engle, 2002) es el estandar de la literatura de pronostico
multivariado financiero, de modo que superarlo -o no lograrlo- dice bastante mas
que superar a un gaussiano estatico.

Este control usa un conjunto de informacion estrictamente mas rico que el resto:
se estima sobre retornos diarios y se proyecta hacia adelante, mientras que los
demas reciben solo los momentos terminales del horizonte. La asimetria es
deliberada y debe declararse al interpretar cualquier comparacion.

Estimacion en dos etapas, como en la formulacion original:

1. GARCH(1,1) por activo con variance targeting y cuasi-verosimilitud gaussiana.
   La QMLE gaussiana es consistente aunque las innovaciones no sean normales
   (Bollerslev y Wooldridge, 1992), por lo que la primera etapa no necesita
   comprometerse con una distribucion de colas.
2. Correlacion condicional dinamica sobre los residuos estandarizados.

La simulacion usa innovaciones t multivariadas cuyos grados de libertad se
estiman de los residuos estandarizados, reutilizando el mismo estimador que
calibra el control Student-t.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from mm_ipsa.models.benchmarks import estimate_student_t_df, nearest_psd

# Con alpha + beta muy cerca de uno la varianza incondicional deja de existir y
# la proyeccion a H pasos se vuelve inestable.
MAX_PERSISTENCE = 0.999


def garch11_variances(
    residuals: np.ndarray,
    alpha: float,
    beta: float,
    unconditional: float,
) -> np.ndarray:
    """Sendero de varianzas condicionales con variance targeting.

    Se parametriza ``omega = sigma2 (1 - alpha - beta)`` para que la varianza
    incondicional del modelo coincida por construccion con la muestral. Esto
    elimina un parametro, evita el modo en que la verosimilitud se aplana cuando
    omega y la persistencia se compensan, y garantiza estacionariedad.

    Devuelve ``T + 1`` valores: el ultimo es la varianza de un paso adelante.
    """
    e = np.asarray(residuals, dtype=float)
    if e.ndim != 1 or len(e) < 2:
        raise ValueError("residuals debe ser un vector de largo al menos dos")
    persistence = alpha + beta
    if alpha < 0 or beta < 0 or persistence >= 1.0:
        raise ValueError("Se requiere alpha, beta >= 0 y alpha + beta < 1")
    if unconditional <= 0:
        raise ValueError("La varianza incondicional debe ser positiva")

    omega = unconditional * (1.0 - persistence)
    variances = np.empty(len(e) + 1, dtype=float)
    variances[0] = unconditional
    for t in range(len(e)):
        variances[t + 1] = omega + alpha * e[t] ** 2 + beta * variances[t]
    return variances


def _decode_persistence(parameters: np.ndarray, ceiling: float) -> tuple[float, float]:
    """Traduce coordenadas de caja a ``(alpha, beta)`` siempre estacionarios.

    Con ``alpha`` y ``beta`` libres, la restriccion ``alpha + beta < 1`` no cabe
    en cotas de caja y el optimizador explora la region infactible. Devolver
    infinito alli destruye el gradiente por diferencias finitas, que se calcula
    como ``inf - inf``, y L-BFGS-B abandona en el punto de arranque.

    Parametrizando por persistencia total y su reparto, la restriccion se cumple
    por construccion y la superficie queda suave en todo el dominio.
    """
    total = ceiling * float(parameters[0])
    split = float(parameters[1])
    return total * split, total * (1.0 - split)


def fit_garch11(returns: np.ndarray) -> dict[str, float]:
    """Ajusta GARCH(1,1) por cuasi-verosimilitud gaussiana con variance targeting."""
    x = np.asarray(returns, dtype=float)
    if x.ndim != 1 or len(x) < 50:
        raise ValueError("Se requieren al menos cincuenta observaciones diarias")
    if not np.isfinite(x).all():
        raise ValueError("returns contiene valores no finitos")

    mean = float(x.mean())
    residuals = x - mean
    unconditional = float(np.var(residuals))
    if unconditional <= 0:
        raise ValueError("La serie no tiene variacion")

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        alpha, beta = _decode_persistence(parameters, MAX_PERSISTENCE)
        variances = garch11_variances(residuals, alpha, beta, unconditional)[:-1]
        if not np.all(variances > 0) or not np.all(np.isfinite(variances)):
            return 1e12
        # Constante omitida: no altera el argmax.
        return 0.5 * float(np.sum(np.log(variances) + residuals**2 / variances))

    best: dict[str, float] | None = None
    # Varios inicios porque la superficie es plana en la direccion de la
    # persistencia y un unico arranque puede quedarse en una meseta.
    for total0, split0 in ((0.95, 0.05), (0.90, 0.12), (0.97, 0.03), (0.80, 0.25)):
        result = minimize(
            negative_log_likelihood,
            np.array([total0, split0]),
            method="L-BFGS-B",
            bounds=[(1e-4, 1.0 - 1e-9), (1e-6, 1.0 - 1e-6)],
            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10},
        )
        if not np.isfinite(result.fun):
            continue
        if best is None or result.fun < best["negative_log_likelihood"]:
            alpha, beta = _decode_persistence(result.x, MAX_PERSISTENCE)
            best = {
                "alpha": alpha,
                "beta": beta,
                "omega": unconditional * (1.0 - alpha - beta),
                "mean": mean,
                "unconditional_variance": unconditional,
                "persistence": alpha + beta,
                "negative_log_likelihood": float(result.fun),
            }
    if best is None:
        raise RuntimeError("El ajuste GARCH(1,1) no convergio desde ningun inicio")
    return best


def _normalise_to_correlation(matrices: np.ndarray) -> np.ndarray:
    """Convierte una pila de matrices Q en correlaciones condicionales R."""
    diagonal = np.sqrt(np.maximum(np.diagonal(matrices, axis1=-2, axis2=-1), 1e-300))
    return matrices / (diagonal[..., :, None] * diagonal[..., None, :])


def _dcc_quasi_log_likelihood_core(
    z: np.ndarray,
    outer_products: np.ndarray,
    unconditional: np.ndarray,
    a: float,
    b: float,
) -> float:
    """Nucleo de la cuasi-verosimilitud con los productos externos ya calculados.

    ``z_t z_t'`` no depende de ``(a, b)``, de modo que recalcularlo en cada
    evaluacion multiplicaba el costo del ajuste sin aportar nada. Se recibe
    precalculado desde el llamador.
    """
    if a < 0 or b < 0 or a + b >= 1.0:
        return -np.inf
    q = unconditional.copy()
    baseline = (1.0 - a - b) * unconditional
    total = 0.0
    for t in range(len(z)):
        scale = np.sqrt(np.maximum(np.diag(q), 1e-300))
        # Con R = D^-1 Q D^-1 se tiene R^-1 = D Q^-1 D, de modo que la forma
        # cuadratica es (D z)' Q^-1 (D z): el vector se MULTIPLICA por la escala.
        # Y log|R| = log|Q| - 2 sum(log s). Asi basta factorizar Q una vez;
        # slogdet mas solve factorizaria dos veces la misma matriz.
        scaled = z[t] * scale
        try:
            cholesky = np.linalg.cholesky(q)
        except np.linalg.LinAlgError:
            return -np.inf
        diagonal = np.diag(cholesky)
        if not np.all(diagonal > 0):
            return -np.inf
        solved = solve_triangular(cholesky, scaled, lower=True, check_finite=False)
        total += (
            2.0 * float(np.sum(np.log(diagonal)))
            - 2.0 * float(np.sum(np.log(scale)))
            + float(solved @ solved)
        )
        q = baseline + a * outer_products[t] + b * q
    return -0.5 * total


def dcc_quasi_log_likelihood(
    standardized: np.ndarray, a: float, b: float
) -> float:
    """Cuasi-verosimilitud de la segunda etapa DCC.

    Se descarta el termino ``z' z``, constante respecto de ``(a, b)``, de modo
    que el valor no es comparable con una verosimilitud completa pero si entre
    parametros.
    """
    z = np.asarray(standardized, dtype=float)
    unconditional = nearest_psd(np.cov(z, rowvar=False, bias=True))
    outer_products = z[:, :, None] * z[:, None, :]
    return _dcc_quasi_log_likelihood_core(z, outer_products, unconditional, a, b)


def fit_dcc(standardized: np.ndarray) -> dict[str, float]:
    """Estima los parametros de correlacion dinamica (a, b) de Engle (2002)."""
    z = np.asarray(standardized, dtype=float)
    if z.ndim != 2 or z.shape[0] < 50 or z.shape[1] < 2:
        raise ValueError("standardized debe ser (T, n) con T >= 50 y n >= 2")
    if not np.isfinite(z).all():
        raise ValueError("standardized contiene valores no finitos")

    # Constantes respecto de (a, b): se calculan una sola vez para todo el ajuste.
    unconditional = nearest_psd(np.cov(z, rowvar=False, bias=True))
    outer_products = z[:, :, None] * z[:, None, :]

    def negative(parameters: np.ndarray) -> float:
        a, b = _decode_persistence(parameters, 0.999)
        value = _dcc_quasi_log_likelihood_core(z, outer_products, unconditional, a, b)
        return -value if np.isfinite(value) else 1e12

    best: dict[str, float] | None = None
    for total0, split0 in ((0.97, 0.02), (0.95, 0.06), (0.93, 0.12)):
        result = minimize(
            negative,
            np.array([total0, split0]),
            method="L-BFGS-B",
            bounds=[(1e-4, 1.0 - 1e-9), (1e-6, 0.5)],
            options={"maxiter": 120, "ftol": 1e-12},
        )
        if not np.isfinite(result.fun):
            continue
        if best is None or result.fun < best["negative_quasi_log_likelihood"]:
            a, b = _decode_persistence(result.x, 0.999)
            best = {
                "a": a,
                "b": b,
                "persistence": a + b,
                "negative_quasi_log_likelihood": float(result.fun),
            }
    if best is None:
        raise RuntimeError("La segunda etapa DCC no convergio")
    return best


def fit_dcc_garch(daily_returns: np.ndarray) -> dict:
    """Ajuste completo en dos etapas sobre una matriz de retornos diarios."""
    x = np.asarray(daily_returns, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("daily_returns debe ser una matriz (T, n) con n >= 2")

    marginals = [fit_garch11(x[:, asset]) for asset in range(x.shape[1])]
    means = np.array([model["mean"] for model in marginals])
    residuals = x - means

    variance_paths = np.column_stack(
        [
            garch11_variances(
                residuals[:, asset],
                marginals[asset]["alpha"],
                marginals[asset]["beta"],
                marginals[asset]["unconditional_variance"],
            )
            for asset in range(x.shape[1])
        ]
    )
    # La ultima fila es la varianza de un paso adelante: el estado desde el que
    # arranca la proyeccion.
    standardized = residuals / np.sqrt(variance_paths[:-1])
    next_variance = variance_paths[-1]

    correlation = fit_dcc(standardized)
    unconditional_q = nearest_psd(np.cov(standardized, rowvar=False, bias=True))

    a, b = correlation["a"], correlation["b"]
    q = unconditional_q.copy()
    for t in range(len(standardized)):
        q = (
            (1.0 - a - b) * unconditional_q
            + a * np.outer(standardized[t], standardized[t])
            + b * q
        )

    # Grados de libertad de las innovaciones, con el mismo estimador que calibra
    # el control Student-t. Los residuos estandarizados tienen media cero y
    # covarianza aproximadamente igual a la incondicional de Q.
    innovation = estimate_student_t_df(
        standardized, np.zeros(x.shape[1]), unconditional_q
    )

    return {
        "marginals": marginals,
        "means": means,
        "standardized": standardized,
        "next_variance": next_variance,
        "last_standardized": standardized[-1],
        "unconditional_q": unconditional_q,
        "next_q": q,
        "a": a,
        "b": b,
        "innovation_df": float(innovation["degrees_of_freedom"]),
        "dcc_persistence": correlation["persistence"],
    }


def simulate_dcc_terminal(
    state: dict,
    horizon: int,
    n_scenarios: int,
    seed: int,
    terminal_mean: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Proyecta el estado condicional H pasos y compone el retorno terminal.

    Todas las trayectorias avanzan en paralelo: la correlacion condicional
    depende del camino, de modo que ``Q`` se mantiene como una pila ``(N, n, n)``
    y las factorizaciones de Cholesky se hacen por lote.

    ``terminal_mean`` recentra la distribucion terminal al mismo primer momento
    que reciben los demas controles. La covarianza NO se impone: reproducirla es
    justamente lo que este modelo hace de forma distinta.
    """
    if horizon < 1 or n_scenarios < 2:
        raise ValueError("horizon debe ser >=1 y n_scenarios >=2")

    rng = np.random.default_rng(seed)
    marginals = state["marginals"]
    n_assets = len(marginals)
    alpha = np.array([model["alpha"] for model in marginals])
    beta = np.array([model["beta"] for model in marginals])
    omega = np.array([model["omega"] for model in marginals])
    means = np.asarray(state["means"], dtype=float)

    a, b = float(state["a"]), float(state["b"])
    unconditional_q = np.asarray(state["unconditional_q"], dtype=float)
    degrees = float(state["innovation_df"])
    if degrees <= 2.0:
        raise ValueError("Los grados de libertad de las innovaciones deben ser > 2")

    variance = np.tile(np.asarray(state["next_variance"], dtype=float), (n_scenarios, 1))
    q = np.tile(np.asarray(state["next_q"], dtype=float), (n_scenarios, 1, 1))
    compounded = np.ones((n_scenarios, n_assets), dtype=float)
    scale = np.sqrt((degrees - 2.0) / degrees)

    for _ in range(horizon):
        correlation = _normalise_to_correlation(q)
        # Un jitter minimo evita que una correlacion casi singular detenga la
        # factorizacion en alguna trayectoria aislada.
        jitter = 1e-10 * np.eye(n_assets)
        cholesky = np.linalg.cholesky(correlation + jitter)

        gaussian = rng.standard_normal((n_scenarios, n_assets))
        correlated = np.einsum("sij,sj->si", cholesky, gaussian)
        chi_square = rng.chisquare(degrees, size=n_scenarios)
        innovations = correlated * scale / np.sqrt(chi_square / degrees)[:, None]

        shocks = np.sqrt(variance) * innovations
        compounded *= 1.0 + (means + shocks)

        variance = omega + alpha * shocks**2 + beta * variance
        q = (
            (1.0 - a - b) * unconditional_q
            + a * innovations[:, :, None] * innovations[:, None, :]
            + b * q
        )

    scenarios = compounded - 1.0
    if terminal_mean is not None:
        scenarios = scenarios - scenarios.mean(axis=0) + np.asarray(terminal_mean)
    return scenarios, np.full(n_scenarios, 1.0 / n_scenarios)


def dcc_garch_terminal(
    daily_returns: np.ndarray,
    horizon: int,
    n_scenarios: int,
    seed: int,
    terminal_mean: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Genera escenarios terminales DCC-GARCH y devuelve el diagnostico del ajuste."""
    state = fit_dcc_garch(daily_returns)
    scenarios, probabilities = simulate_dcc_terminal(
        state, horizon, n_scenarios, seed, terminal_mean
    )
    diagnostics = {
        "dcc_a": state["a"],
        "dcc_b": state["b"],
        "dcc_persistence": state["dcc_persistence"],
        "innovation_df": state["innovation_df"],
        "mean_garch_alpha": float(np.mean([m["alpha"] for m in state["marginals"]])),
        "mean_garch_beta": float(np.mean([m["beta"] for m in state["marginals"]])),
        "mean_garch_persistence": float(
            np.mean([m["persistence"] for m in state["marginals"]])
        ),
        "max_garch_persistence": float(
            np.max([m["persistence"] for m in state["marginals"]])
        ),
    }
    return scenarios, probabilities, diagnostics
