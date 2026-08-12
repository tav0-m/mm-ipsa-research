# Protocolo de investigación

## 1. Objetivo y unidad de análisis

El objetivo es comparar generadores de distribución de retornos terminales de cinco días para 15 acciones chilenas y medir su efecto sobre decisiones long-only. La unidad temporal de evaluación es una ventana H=5 no solapada; la estimación puede usar ventanas rolling, dejando explícita su dependencia.

## 2. Hipótesis falsables

- H1: MM reduce el CRPS medio frente al Gaussian terminal comparable.
- H2: MM reduce Energy Score o Variogram Score sin deteriorar materialmente el otro.
- H3: la tasa de excepciones VaR al 5% es compatible con 5% y los hits no muestran clustering según Kupiec/Christoffersen.
- H4: un portafolio derivado de MM mejora Sharpe neto o CVaR frente a Equal Weight y baselines robustos; el IC95 bootstrap del diferencial debe excluir cero para una afirmación fuerte.

Si estas condiciones no se cumplen, la conclusión válida es que MM no demostró ventaja en esta muestra.

## 3. Separación temporal

- Entrenamiento inicial: hasta 2023-12-31.
- Validación de desarrollo: desde 2024-01-02. Ya fue observada y no puede llamarse test final.
- Test confirmatorio futuro: comienza solo después de congelar versión, universo, hipótesis, scores primarios e hiperparámetros. Su fecha todavía no se fija porque depende de la fecha de congelamiento.

No se permite elegir `entropy_lambda`, vida media, shrinkage, número de escenarios o estrategia usando el test confirmatorio.

Como prueba de estabilidad retrospectiva se define `rolling_origin_expanding_v1`. Usa cuatro folds explícitos con evaluación en 2023, 2024, 2025 y 2026-H1. En cada origen se reconstruyen los targets y se recalibran MM y los tres controles usando exclusivamente observaciones anteriores. Las ventanas de evaluación son disjuntas entre folds y se conservan en `research/rolling_origin.yaml`.

## 4. Datos y sesgos

1. Guardar precios raw y disponibilidad antes de imputar.
2. Fallar si la cobertura raw cae bajo el umbral configurado.
3. Limitar `forward fill`; no usar `backfill`.
4. Calcular retornos con `fill_method=None`.
5. Reportar retornos cero como proxy de iliquidez/precio stale, sin asumir automáticamente que son errores.
6. Mantener universo fijo para el experimento actual y declarar que esto no elimina survivorship bias.
7. Yahoo Finance es una fuente conveniente, no un feed institucional; una fase posterior debe reconciliar eventos corporativos con una segunda fuente.

## 5. Modelo primario

La distribución discreta MM minimiza una función propia:

\[
F(X,p)=\sum_{k=1}^{4}\sum_i W_{ki}(\hat m_{ki}-M_{ki})^2
+w_\Sigma\sum_{i\ne j}A_{ij}(\hat\Sigma_{ij}-\Sigma_{ij})^2.
\]

La diagonal de covarianza se excluye porque la varianza ya aparece en el segundo momento. Las escalas se basan en potencias de volatilidad para evitar pesos explosivos cuando la media o el tercer momento se aproximan a cero.

El solver optimiza:

\[
G(X,p)=F(X,p)+\lambda_{KL}D_{KL}(p\|u).
\]

La convergencia y la selección multi-start se realizan sobre G. El bloque completo de posiciones X se optimiza conjuntamente con L-BFGS, porque las covarianzas acoplan columnas y destruyen la separabilidad del Jacobi paralelo.

## 6. Baselines justos

Todos reciben la misma media, covarianza, horizonte H y ponderación temporal:

- Gaussian terminal multivariado.
- Student-t terminal con grados de libertad prefijados y covarianza parametrizada.
- Distribución histórica EWMA sin ruido Monte Carlo.

Wiener diario queda como denominación legacy, no como benchmark principal.

## 7. Sensibilidad de liquidez

El universo de 15 activos permanece como análisis principal. La sensibilidad `liquidity_is_filter_v1` se define en un archivo separado y solo puede usar métricas calculadas hasta 2023-12-31: tasa de retornos cero IS no mayor a 3% y racha máxima IS no mayor a cinco sesiones. Debe conservar al menos ocho activos.

Las métricas OOS pueden mostrarse como diagnóstico después de seleccionar, pero no pueden participar directa ni indirectamente en la inclusión. Cada ejecución debe recalibrar todos los modelos con los mismos hiperparámetros, restricciones, semillas y costos del análisis principal. Los niveles de score entre universos de distinta dimensión no se interpretan como una comparación causal; se prioriza el ranking interno y la estabilidad de las conclusiones.

## 8. Evaluación

Métricas primarias: CRPS marginal y Energy Score multivariado. Secundarias: Variogram Score, error de correlación, VaR/ES y pruebas de Kupiec/Christoffersen. Las comparaciones se realizan sobre ventanas OOS no solapadas.

Los scores se conservan por ventana y se comparan como pérdidas pareadas `MM − control`; valores negativos favorecen a MM. El contraste primario registrado es CRPS de MM contra Gaussian terminal. La dependencia remanente se trata con moving-block bootstrap de cuatro ventanas H=5, 5.000 remuestreos e IC95 básico centrado. Se aplica Holm conjuntamente a los nueve contrastes formados por tres scores y tres controles, por separado en el universo completo y en la sensibilidad líquida.

En rolling-origin, el remuestreo mantiene cada bloque dentro de su fold: ningún bloque puede concatenar el final de un año con el inicio de otro. La media pooled pondera observaciones, mientras que el resumen de estabilidad reporta también rango y número de victorias por fold. Los hiperparámetros permanecen fijos entre folds.

Portafolios: Equal Weight, inverse variance, HRP, mínima varianza regularizada, mínimo CVaR y máximo Sharpe regularizado. El backtest incluye deriva de pesos, turnover y costos explícitos. Los IC se calculan mediante moving-block bootstrap.

## 9. Regla de publicación

Toda tabla debe identificar: hash de código, configuración, fecha UTC, fuentes, rango de datos, estado de evaluación y si el resultado es exploratorio o confirmatorio. No se publican solo los mejores starts o hiperparámetros; se conservan todos los resultados relevantes.

Una diferencia se interpreta como evidencia en esta validación únicamente si su IC excluye cero y el p-valor ajustado de Holm es menor a 5%. Incluso entonces no se etiqueta confirmatoria porque el periodo ya fue observado.
