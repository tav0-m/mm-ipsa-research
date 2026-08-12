"""Optimizador multi-start por descenso en bloques para escenarios MM.

La función de selección es ``G = F + lambda * KL(p || uniforme)``. El bloque
de probabilidades usa mirror descent entrópico por defecto y el bloque de
escenarios optimiza X completo con L-BFGS-B, porque la covarianza acopla los
activos. Los modos SLSQP, secuencial y Jacobi se conservan únicamente como
alternativas explícitas para experimentación y compatibilidad.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import warnings as _warnings
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ObjectiveProtocol(Protocol):
    @property
    def N(self) -> int: ...

    @property
    def n(self) -> int: ...

    @property
    def M(self) -> np.ndarray: ...

    def evaluate(self, x: np.ndarray, p: np.ndarray) -> float: ...

    def grad_x(self, x: np.ndarray, p: np.ndarray) -> np.ndarray: ...

    def grad_p(self, x: np.ndarray, p: np.ndarray) -> np.ndarray: ...


# Funcion libre para que pickle pueda serializarla (requerido por Pool)
def _optimize_column(args):
    obj, x_snapshot, p, i = args

    def fun(xi):
        xt = x_snapshot.copy()
        xt[:, i] = xi
        return obj.evaluate(xt, p)

    def grad(xi):
        xt = x_snapshot.copy()
        xt[:, i] = xi
        return obj.grad_x(xt, p)[:, i]
    res = minimize(fun, x_snapshot[:, i], jac=grad, method='L-BFGS-B',
                   options={'ftol': 1e-9, 'maxiter': 100, 'maxls': 20})
    return i, res.x


class BCDSolver:
    """
    Solver BCD multi-start con bloque X conjunto y probabilidades en simplex.

    Parameters
    ----------
    objective  : MMObjective
    cfg_mm     : dict con N_scenarios, bcd_max_iter, tol, min_iter, n_starts, seed
    n_workers  : int  nucleos para Pool (None = detectar automaticamente)
    warm_noise : float  amplitud del warm restart como fraccion de sigma_hist
    """

    def __init__(self, objective: ObjectiveProtocol, cfg_mm: dict,
                 n_workers: int | None = None, warm_noise: float = 0.15):
        self.obj            = objective
        self.N              = cfg_mm["N_scenarios"]
        self.n              = objective.n
        self.max_iter       = cfg_mm["bcd_max_iter"]
        self.tol            = cfg_mm["tol"]
        self.min_iter       = cfg_mm.get("min_iter", 5)
        self.n_starts       = cfg_mm["n_starts"]
        self.seed           = cfg_mm["seed"]
        self.warm_noise     = warm_noise
        self.early_stop_F   = cfg_mm.get("early_stop_F", None)
        self.update_mode    = str(cfg_mm.get("update_mode", "sequential"))
        self.x_inner_max_iter = int(cfg_mm.get("x_inner_max_iter", 100))
        self.x_ftol         = float(cfg_mm.get("x_ftol", 1e-12))
        self.x_gtol         = float(cfg_mm.get("x_gtol", 1e-7))
        self.x_maxls        = int(cfg_mm.get("x_maxls", 40))
        self.start_strategy = str(cfg_mm.get("start_strategy", "independent"))
        self.p_solver       = str(cfg_mm.get("p_solver", "mirror_descent"))
        self.p_inner_max_iter = int(cfg_mm.get("p_inner_max_iter", 50))
        self.p_step_scale   = float(cfg_mm.get("p_step_scale", 1.0))
        self.p_inner_tol    = float(cfg_mm.get("p_inner_tol", 1e-8))
        self.patience       = int(cfg_mm.get("convergence_patience", 3))
        self.strict_solver  = bool(cfg_mm.get("strict_solver", True))
        self.acceptance_tol = float(cfg_mm.get("acceptance_tol", 1e-10))
        self.x_stationarity_tol = float(cfg_mm.get("x_stationarity_tol", np.inf))
        self.p_stationarity_tol = float(cfg_mm.get("p_stationarity_tol", np.inf))
        if self.update_mode not in {"joint_lbfgs", "sequential", "parallel_jacobi"}:
            raise ValueError("update_mode debe ser joint_lbfgs, sequential o parallel_jacobi")
        if self.start_strategy not in {"independent", "warm"}:
            raise ValueError("start_strategy debe ser independent o warm")
        if self.p_solver not in {"mirror_descent", "slsqp"}:
            raise ValueError("p_solver debe ser mirror_descent o slsqp")
        if self.p_inner_max_iter <= 0 or self.p_step_scale <= 0:
            raise ValueError("Parametros del solver-p no validos")
        if self.x_inner_max_iter <= 0 or self.x_ftol <= 0 or self.x_gtol <= 0 or self.x_maxls <= 0:
            raise ValueError("Parametros del solver-x no validos")
        if self.patience < 1:
            raise ValueError("convergence_patience debe ser >= 1")

        # Regularización KL respecto de la distribución uniforme.
        self.entropy_lambda = float(cfg_mm.get("entropy_lambda", 0.0))

        available       = mp.cpu_count()
        self.n_workers  = min(n_workers if n_workers else available, self.n)

        # Sigma historica para warm restart
        self.sigma_hist = np.sqrt(np.abs(objective.M[1]))  # (n,)

        # Resultados
        self.best_x: np.ndarray | None = None
        self.best_p: np.ndarray | None = None
        self.best_F     = np.inf
        self.best_G     = np.inf
        self.best_start_id: int | None = None
        self.history    = []
        self.history_G  = []
        self.all_starts = []
        self.solver_events = []

    # -----------------------------------------------------------------------
    def _init_x(self, rng, warm=False):
        """
        Inicializa x.
        warm=False: muestrea N(mu_hist, sigma_hist)  [primer start]
        warm=True:  x_best + ruido_pequeno           [starts 2..N]
        """
        mu  = self.obj.M[0]
        sig = self.sigma_hist

        if not warm or self.best_x is None:
            return rng.normal(loc=mu[np.newaxis, :],
                              scale=sig[np.newaxis, :],
                              size=(self.N, self.n))
        else:
            noise = rng.normal(0, 1, size=(self.N, self.n))
            return self.best_x + self.warm_noise * sig[np.newaxis, :] * noise

    def _init_p(self, rng):
        raw = rng.exponential(1.0, self.N)
        return raw / raw.sum()

    # -----------------------------------------------------------------------
    def _entropy(self, p: np.ndarray) -> float:
        """
        Entropia de Shannon H(p) = -sum_j p_j * log(p_j).

        Se usa un clip en 1e-300 para evitar log(0) = -inf.
        Cuando p_j -> 0, el termino p_j * log(p_j) -> 0 por L'Hopital,
        por lo que el clip no distorsiona el valor real de la entropia.
        """
        p_safe = np.where(p > 0, p, 1e-300)
        return -float(np.sum(p * np.log(p_safe)))

    def _entropy_grad(self, p: np.ndarray) -> np.ndarray:
        """
        Gradiente de la entropia: dH/dp_j = -(log(p_j) + 1).

        El gradiente del termino regularizador -lambda*H(p) es:
            d(-lambda*H)/dp_j = lambda * (log(p_j) + 1)

        Esto penaliza probabilidades pequenas (log(p_j) << 0 -> gradiente
        muy negativo sobre F), empujando la distribucion hacia el uniforme.
        """
        p_safe = np.where(p > 0, p, 1e-300)
        return np.log(p_safe) + 1.0   # gradiente de H; se multiplica por -lambda afuera

    def _kl_uniform(self, p: np.ndarray) -> float:
        """KL(p || uniforme), no negativa y comparable entre corridas."""
        p_safe = np.clip(np.asarray(p, dtype=float), 1e-300, None)
        return float(np.sum(p_safe * np.log(p_safe * self.N)))

    def regularized_objective(self, x: np.ndarray, p: np.ndarray) -> float:
        """G(x,p) = F(x,p) + lambda * KL(p || uniforme)."""
        return self.obj.evaluate(x, p) + self.entropy_lambda * self._kl_uniform(p)

    def stationarity_metrics(self, x: np.ndarray, p: np.ndarray) -> dict[str, float]:
        """Residuos de primer orden para X libre y p interior al simplex."""
        gradient_x = self.obj.grad_x(x, p)
        gradient_p = self.obj.grad_p(x, p).copy()
        if self.entropy_lambda > 0:
            gradient_p += self.entropy_lambda * (
                np.log(np.clip(p * self.N, 1e-300, None)) + 1.0
            )
        # En un óptimo interior sujeto a sum(p)=1, grad_p es constante.
        tangent_p = gradient_p - float(np.mean(gradient_p))
        return {
            "x_gradient_l2": float(np.linalg.norm(gradient_x)),
            "x_gradient_inf": float(np.max(np.abs(gradient_x))),
            "p_tangent_gradient_l2": float(np.linalg.norm(tangent_p)),
            "p_tangent_gradient_inf": float(np.max(np.abs(tangent_p))),
        }

    def _step_p_slsqp(self, x, p):
        """Alternativa SLSQP para minimizar G sobre el simplex.

        Se conserva para comparación. No se presume convexidad global de F
        ni se afirma convergencia del algoritmo; el candidato solo se acepta
        cuando reduce el objetivo regularizado.
        """
        lam = self.entropy_lambda
        obj = self.obj

        def objective_reg(p_):
            return obj.evaluate(x, p_) + lam * self._kl_uniform(p_)

        def gradient_reg(p_):
            g = obj.grad_p(x, p_).copy()
            if lam > 0.0:
                # d KL(p||u)/dp = log(p*N)+1. El termino constante log(N)
                # es irrelevante sobre el plano sum(p)=1, pero se incluye para
                # que gradiente y objetivo representen exactamente la misma G.
                p_safe = np.clip(p_, 1e-300, None)
                g += lam * (np.log(p_safe * self.N) + 1.0)
            return g

        # p0 interior para evitar log(0) en el primer gradiente
        # FIX lb: 1e-12 → 1e-6 para evitar gradiente de entropia extremo.
        # Con lam=0.001, el equilibrio natural es p* ~ 1/N = 1/500 = 0.002,
        # muy por encima de 1e-6. El bound 1e-12 causaba que el gradiente de
        # entropia en p_j=1e-12 fuera log(1e-12)+1 ≈ -26.6, forzando pasos
        # muy grandes que salen de los bounds → RuntimeWarning de SciPy.
        lb = 1e-6 if lam > 0.0 else 0.0

        # p0: clip al bound lb más un margen para empezar en el interior
        # Usamos max(lb, 1e-5) como piso del punto de inicio para que SLSQP
        # no parta desde la frontera (reduce iteraciones internas).
        p0_floor = max(lb, 1e-5) if lam > 0.0 else 0.0
        p0 = np.maximum(p, p0_floor)
        p0 /= p0.sum()

        # Suprimir el RuntimeWarning de SciPy sobre bounds clipping:
        # es informacional (SLSQP recorta y continúa correctamente).
        with _warnings.catch_warnings():
            _warnings.filterwarnings(
                "ignore", category=RuntimeWarning,
                message="Values in x were outside bounds"
            )
            result = minimize(
                objective_reg,
                p0,
                jac         = gradient_reg,
                method      = "SLSQP",
                bounds      = [(lb, 1.0)] * self.N,
                constraints = [{"type": "eq", "fun": lambda p_: p_.sum() - 1.0}],
                # FIX maxiter: 300→100 — con N=500 cada iter SLSQP es O(N²).
                # La convergencia del BCD viene de los outer iters, no del sub-solver.
                options     = {"ftol": 1e-8, "maxiter": 100}
            )
        if not result.success:
            msg = f"SLSQP paso-p fallo: status={result.status}, mensaje={result.message}"
            if self.strict_solver:
                raise RuntimeError(msg)
            print(f"    [warn] {msg}; se conserva p anterior")
            return p.copy()
        p_new = np.maximum(result.x, lb)
        p_new /= p_new.sum()
        if self.regularized_objective(x, p_new) > self.regularized_objective(x, p) + self.acceptance_tol:
            msg = "SLSQP paso-p aumento el objetivo regularizado"
            if self.strict_solver:
                raise RuntimeError(msg)
            print(f"    [warn] {msg}; se conserva p anterior")
            return p.copy()
        return p_new

    def _step_p_mirror(self, x, p):
        """Mirror descent entropico sobre el simplex con backtracking.

        La actualizacion exponencial preserva positividad y presupuesto sin un
        SQP denso de dimension N. Cada iteracion interna solo se acepta si
        reduce G, manteniendo el contrato monotono del BCD.
        """
        lam = self.entropy_lambda
        p_current = np.clip(np.asarray(p, dtype=float), 1e-15, None)
        p_current /= p_current.sum()
        G_current = self.regularized_objective(x, p_current)

        for _ in range(self.p_inner_max_iter):
            gradient = self.obj.grad_p(x, p_current).copy()
            if lam > 0:
                gradient += lam * (
                    np.log(np.clip(p_current * self.N, 1e-300, None)) + 1.0
                )
            # Una constante en el gradiente es irrelevante sobre el simplex.
            gradient -= float(p_current @ gradient)
            scale = max(float(np.max(np.abs(gradient))), 1e-12)
            step = self.p_step_scale / scale
            accepted = False
            candidate = p_current
            G_candidate = G_current

            for _ in range(25):
                logits = np.log(np.clip(p_current, 1e-300, None)) - step * gradient
                logits -= logits.max()
                candidate = np.exp(logits)
                candidate /= candidate.sum()
                G_candidate = self.regularized_objective(x, candidate)
                if (
                    np.isfinite(G_candidate)
                    and G_candidate <= G_current + self.acceptance_tol
                ):
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                if self.strict_solver and not np.isfinite(G_current):
                    raise RuntimeError("Mirror descent no encontro un punto finito")
                break
            relative = abs(G_current - G_candidate) / (abs(G_current) + 1e-12)
            p_current, G_current = candidate, G_candidate
            if relative < self.p_inner_tol:
                break
        return p_current

    def _step_p(self, x, p):
        """Despacha al solver de probabilidades configurado."""
        if self.p_solver == "slsqp":
            return self._step_p_slsqp(x, p)
        return self._step_p_mirror(x, p)

    # -----------------------------------------------------------------------
    def _step_x_sequential(self, x, p):
        """BCD Gauss-Seidel: actualiza una columna y valida descenso de G."""
        obj   = self.obj
        x_new = x.copy()
        for i in range(self.n):
            G_before = self.regularized_objective(x_new, p)

            def fun_i(xi, i=i):
                xt = x_new.copy()
                xt[:, i] = xi
                return obj.evaluate(xt, p)

            def grad_i(xi, i=i):
                xt = x_new.copy()
                xt[:, i] = xi
                return obj.grad_x(xt, p)[:, i]
            res = minimize(fun_i, x_new[:, i], jac=grad_i,
                           method="L-BFGS-B",
                           options={
                               "ftol": self.x_ftol,
                               "gtol": self.x_gtol,
                               "maxiter": self.x_inner_max_iter,
                               "maxls": self.x_maxls,
                           })
            candidate = x_new.copy()
            candidate[:, i] = res.x
            G_after = self.regularized_objective(candidate, p)
            if not res.success:
                self.solver_events.append({
                    "block": "x",
                    "asset": i,
                    "status": int(res.status),
                    "message": str(res.message),
                    "accepted_by_descent": bool(
                        np.isfinite(G_after)
                        and G_after <= G_before + self.acceptance_tol
                    ),
                })
            if np.isfinite(G_after) and G_after <= G_before + self.acceptance_tol:
                x_new = candidate
            elif self.strict_solver and not np.all(np.isfinite(res.x)):
                raise RuntimeError(
                    f"Paso-x activo={i} produjo una solucion no finita"
                )
        return x_new

    def _step_x_parallel(self, x, p):
        """Actualización Jacobi aproximada de columnas en paralelo.

        La covarianza acopla activos, por lo que las columnas no son
        separables. Cada subproblema usa el mismo snapshot y el resultado
        conjunto no posee garantía de descenso; por eso no es el modo por
        defecto y queda disponible solo para experimentos comparativos.
        """
        x_snap = x.copy()
        args   = [(self.obj, x_snap, p, i) for i in range(self.n)]

        with mp.Pool(processes=self.n_workers) as pool:
            results = pool.map(_optimize_column, args)

        x_new = x.copy()
        for i, xi_opt in results:
            x_new[:, i] = xi_opt
        return x_new

    def _step_x_joint(self, x, p):
        """Optimiza X completo; respeta exactamente el acoplamiento entre activos."""
        shape = x.shape
        G_before = self.regularized_objective(x, p)

        def objective_flat(flat):
            return self.obj.evaluate(flat.reshape(shape), p)

        def gradient_flat(flat):
            return self.obj.grad_x(flat.reshape(shape), p).ravel()

        result = minimize(
            objective_flat,
            x.ravel(),
            jac=gradient_flat,
            method="L-BFGS-B",
            options={
                # SciPy compara ftol con max(|F_k|, |F_{k+1}|, 1). Para
                # objetivos MM << 1, 1e-9 era una tolerancia absoluta demasiado
                # laxa y podía declarar convergencia tras una sola iteración.
                "ftol": self.x_ftol,
                "gtol": self.x_gtol,
                "maxiter": self.x_inner_max_iter,
                "maxls": self.x_maxls,
            },
        )
        candidate = result.x.reshape(shape)
        G_after = self.regularized_objective(candidate, p)
        if not result.success:
            self.solver_events.append({
                "block": "x_joint",
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "gradient_inf": float(np.max(np.abs(result.jac))),
                "accepted_by_descent": bool(
                    np.isfinite(G_after) and G_after <= G_before + self.acceptance_tol
                ),
            })
        if np.isfinite(G_after) and G_after <= G_before + self.acceptance_tol:
            return candidate
        if self.strict_solver and not np.all(np.isfinite(candidate)):
            raise RuntimeError("El bloque X conjunto produjo valores no finitos")
        return x.copy()

    def _step_x(self, x, p):
        """Despacha al modo configurado; joint_lbfgs es el modo principal."""
        if self.update_mode == "joint_lbfgs":
            return self._step_x_joint(x, p)
        if self.update_mode == "parallel_jacobi" and self.n_workers > 1:
            try:
                candidate = self._step_x_parallel(x, p)
                if self.regularized_objective(candidate, p) <= self.regularized_objective(x, p) + self.acceptance_tol:
                    return candidate
                print("    [warn] Jacobi paralelo rechazado por aumento de G; usando secuencial")
                return self._step_x_sequential(x, p)
            except Exception as e:
                print(f"    aviso: Pool fallo ({type(e).__name__}), usando secuencial")
                return self._step_x_sequential(x, p)
        return self._step_x_sequential(x, p)

    # -----------------------------------------------------------------------
    def _run_single_start(self, start_id, seed, use_warm=False):
        rng = np.random.default_rng(seed)
        x   = self._init_x(rng, warm=use_warm)
        p   = self._init_p(rng)
        F   = self.obj.evaluate(x, p)
        G   = self.regularized_objective(x, p)
        F0  = F
        G0  = G

        history        = [F]
        history_G      = [G]
        rel_history    = []      # track cambios relativos por iteración
        converged      = False
        stop_reason    = "max_iter"
        iters          = 0
        stable_iters   = 0

        for it in range(self.max_iter):
            iters = it + 1
            p     = self._step_p(x, p)
            x     = self._step_x(x, p)
            F_new = self.obj.evaluate(x, p)
            G_new = self.regularized_objective(x, p)
            if G_new > G + self.acceptance_tol:
                raise RuntimeError(
                    f"Iteracion BCD aumento G: {G:.6g} -> {G_new:.6g}"
                )
            rel   = abs(G - G_new) / (abs(G) + 1e-10)
            F     = F_new
            G     = G_new
            history.append(F)
            history_G.append(G)
            rel_history.append(rel)

            stable_iters = stable_iters + 1 if rel < self.tol else 0
            if iters >= self.min_iter:
                if stable_iters >= self.patience:
                    converged   = True
                    stop_reason = f"tol_patience_{self.patience}"
                    break

        stationarity = self.stationarity_metrics(x, p)
        stationarity_pass = bool(
            stationarity["x_gradient_inf"] <= self.x_stationarity_tol
            and stationarity["p_tangent_gradient_inf"] <= self.p_stationarity_tol
        )
        return {"start_id": start_id, "seed": seed, "warm": use_warm,
                "F_ini": F0, "F_fin": F, "G_ini": G0, "G_fin": G,
                "iters": iters,
                "converged": converged, "stop_reason": stop_reason,
                "rel_final": rel_history[-1] if rel_history else float("nan"),
                "x": x, "p": p, "history": history, "history_G": history_G,
                "rel_history": rel_history, "stationarity": stationarity,
                "stationarity_pass": stationarity_pass}

    # -----------------------------------------------------------------------
    def solve(self, out_path=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Ejecuta n_starts y devuelve la mejor solucion.

        Estrategia:
          Start 1 : inicio frio (aleatorio)  — explora globalmente
          Start 2+: inicio calido (warm)     — refina vecindad del optimo actual

        Parameters
        ----------
        out_path : str/Path  directorio para guardar objective_history.csv
        """
        print("\n[bcd] multi-start")
        print(f"  starts={self.n_starts}, N={self.N}, n={self.n}, max_iter={self.max_iter}")
        print(f"  update_mode={self.update_mode}, start_strategy={self.start_strategy}")
        print(f"  p_solver={self.p_solver}, p_inner_max_iter={self.p_inner_max_iter}")
        print(f"  patience={self.patience}, strict_solver={self.strict_solver}")
        reg_str = f"entropy_lambda={self.entropy_lambda:.2e}" if self.entropy_lambda > 0 else "sin regularizacion (lambda=0)"
        print(f"  regularizacion={reg_str}")

        t0 = time.time()

        for s in range(self.n_starts):
            seed_s   = self.seed + s * 1000
            use_warm = self.start_strategy == "warm" and s > 0
            tag      = "calido" if use_warm else "frio"

            print(f"  start {s+1}/{self.n_starts}: seed={seed_s}, modo={tag}", flush=True)

            result = self._run_single_start(s + 1, seed_s, use_warm)
            self.all_starts.append(result)

            crit = result.get("stop_reason", "tol" if result["converged"] else "max_iter")
            rel_f = result.get("rel_final", float("nan"))
            eligible = bool(result["converged"] and result["stationarity_pass"])
            flag = " [best-valid]" if eligible and result["G_fin"] < self.best_G else ""
            stationarity_tag = "stationary" if result["stationarity_pass"] else "nonstationary"
            print(f"    F0={result['F_ini']:.4f}, F_final={result['F_fin']:.6f}, "
                  f"G_final={result['G_fin']:.6f}, "
                  f"iters={result['iters']}, stop={crit}, rel={rel_f:.2e}, "
                  f"{stationarity_tag}{flag}")

            if eligible and result["G_fin"] < self.best_G:
                self.best_F    = result["F_fin"]
                self.best_G    = result["G_fin"]
                self.best_x    = result["x"].copy()
                self.best_p    = result["p"].copy()
                self.best_start_id = int(result["start_id"])
                self.history   = result["history"]
                self.history_G = result["history_G"]

            if self.early_stop_F is not None and self.best_F <= float(self.early_stop_F):
                print(f"    early_stop_F alcanzado: {self.best_F:.6g} <= {float(self.early_stop_F):.6g}")
                break

        elapsed  = time.time() - t0
        if self.best_x is None or self.best_p is None:
            message = (
                "Ningun start convergente cumple los umbrales de estacionariedad "
                f"X<={self.x_stationarity_tol:.2e}, p<={self.p_stationarity_tol:.2e}"
            )
            if self.strict_solver:
                raise RuntimeError(message)
            fallback = min(self.all_starts, key=lambda record: record["G_fin"])
            print(f"  [warn] {message}; se usa menor G como fallback no estricto")
            self.best_F = fallback["F_fin"]
            self.best_G = fallback["G_fin"]
            self.best_x = fallback["x"].copy()
            self.best_p = fallback["p"].copy()
            self.best_start_id = int(fallback["start_id"])
            self.history = fallback["history"]
            self.history_G = fallback["history_G"]
        best_x = self.best_x
        best_p = self.best_p
        if best_x is None or best_p is None:
            raise RuntimeError("El solver termino sin una solucion seleccionada")
        n_active  = int((best_p > 1e-6).sum())
        N_eff     = 1.0 / (best_p ** 2).sum()
        entropy   = self._entropy(best_p)
        max_entr  = float(np.log(self.N))
        entr_pct  = entropy / max_entr * 100.0

        # Cuantos escenarios concentran el 50% y 80% de la masa
        p_sorted  = np.sort(best_p)[::-1]
        cumsum    = np.cumsum(p_sorted)
        n50 = int(np.searchsorted(cumsum, 0.50)) + 1
        n80 = int(np.searchsorted(cumsum, 0.80)) + 1

        print("\n[bcd] resumen")
        print(f"  F_best={self.best_F:.6f}, G_best={self.best_G:.6f}, tiempo={elapsed:.1f}s")
        print(f"  activos={n_active}/{self.N}, N_eff={N_eff:.1f}, p_max={best_p.max():.4f}")
        print(f"  entropia={entropy:.4f}/{max_entr:.4f} ({entr_pct:.1f}%), masa50={n50}, masa80={n80}")
        if self.solver_events:
            accepted = sum(event.get("accepted_by_descent", False) for event in self.solver_events)
            print(f"  solver_events={len(self.solver_events)}, aceptados_por_descenso={accepted}")

        if out_path is not None:
            self._save_history(Path(out_path))

        return best_x, best_p


    # -----------------------------------------------------------------------
    def summary_table(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame con el resumen de todos los starts del BCD.

        Columnas:
            start_id  : número de inicio (1-based)
            seed      : semilla usada
            warm      : True si fue warm restart
            F_ini     : valor objetivo inicial
            F_fin     : valor objetivo final (F*)
            iters     : iteraciones ejecutadas
            stop      : criterio de parada ('tol' o 'max_iter')
            rel_final : cambio relativo final |F_t - F_{t-1}| / |F_{t-1}|
            is_best   : True si este start produjo el mejor F*
        """
        if not self.all_starts:
            return pd.DataFrame()

        selected_id = self.best_start_id
        rows = []
        for r in self.all_starts:
            rows.append({
                "start_id" : r["start_id"],
                "seed"     : r["seed"],
                "warm"     : r["warm"],
                "F_ini"    : round(float(r["F_ini"]), 6),
                "F_fin"    : round(float(r["F_fin"]), 8),
                "G_ini"    : round(float(r["G_ini"]), 6),
                "G_fin"    : round(float(r["G_fin"]), 8),
                "iters"    : r["iters"],
                "stop"     : r.get("stop_reason", "tol" if r["converged"] else "max_iter"),
                "rel_final": float(r.get("rel_final", float("nan"))),
                "x_gradient_inf": r["stationarity"]["x_gradient_inf"],
                "p_tangent_gradient_inf": r["stationarity"]["p_tangent_gradient_inf"],
                "stationarity_pass": r["stationarity_pass"],
                "is_best"  : r["start_id"] == selected_id,
            })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    def _save_history(self, out_dir):
        """Guarda objective_history.csv para generate_all_plots.py."""
        histories = {f"start_{r['start_id']}": r["history"]
                     for r in self.all_starts if r.get("history")}
        if not histories:
            return
        max_len = max(len(h) for h in histories.values())
        rows = {}
        for name, hist in histories.items():
            padded = hist + [hist[-1]] * (max_len - len(hist))
            rows[name] = padded
        df = pd.DataFrame(rows)
        df.index.name = "iter"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "objective_history.csv")
        histories_g = {f"start_{r['start_id']}": r["history_G"]
                       for r in self.all_starts if r.get("history_G")}
        max_len_g = max(len(h) for h in histories_g.values())
        rows_g = {
            name: hist + [hist[-1]] * (max_len_g - len(hist))
            for name, hist in histories_g.items()
        }
        df_g = pd.DataFrame(rows_g)
        df_g.index.name = "iter"
        df_g.to_csv(out_dir / "regularized_objective_history.csv")
        print(f"  objective_history.csv guardado en {out_dir}/")
