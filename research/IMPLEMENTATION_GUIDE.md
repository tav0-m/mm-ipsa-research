# Guía paso a paso de las mejoras implementadas

## 1. Corregir el núcleo matemático

**Problema:** el gradiente del término de covarianza tenía la mitad del valor correcto. El bloque paralelo suponía separabilidad pese al acoplamiento de covarianzas y el criterio de parada observaba F aunque el paso de probabilidades optimizaba F más entropía.

**Implementación:** `src/mm/objective.py` corrige factores analíticos; `src/mm/bcd.py` define G=F+lambda KL, valida descenso por bloque, optimiza X conjuntamente para respetar covarianzas, usa starts independientes y paciencia de convergencia. El bloque X usa `x_ftol=1e-12`: SciPy compara esta tolerancia con `max(|F|,1)`, por lo que el valor anterior `1e-9` detenía prematuramente objetivos del orden de `1e-7`. También se reportan residuos de primer orden para X y para el gradiente tangente de p.

**Verificación:**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_objective tests.test_bcd -v
```

Se esperan diferencias finitas dentro de tolerancia y una historia G no creciente.

## 2. Dar significado correcto al EWMA

**Problema:** `lambda=0.94` aplicado a filas diarias tiene vida media de 11.2 observaciones, no 11 semanas.

**Implementación:** `ewma_half_life_weeks=11` se transforma en `0.5^(1/55)`. Se registra N efectivo y se contrae la covarianza 10% hacia su diagonal como control transparente, no como Ledoit-Wolf exacto.

**Verificación:**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_targets -v
```

La prueba exige `lambda^55=0.5`.

## 3. Endurecer la ingestión

**Problema:** la cobertura se medía después de imputar, ocultando huecos.

**Implementación:** se guardan raw, máscara, cobertura, primer/último dato, máximo hueco, imputaciones, parámetros de la solicitud, versión del proveedor y hashes internos. Las escrituras son atómicas y el forward fill está limitado. La cobertura insuficiente detiene el pipeline sin reemplazar silenciosamente el panel canónico.

**Ejecución y evidencia:**

```powershell
.\.venv\Scripts\python.exe run.py --step download
Import-Csv .\outputs\data_quality_prices.csv | Format-Table
Get-Content .\outputs\data_download_metadata.json
```

La transformación usa la máscara raw. Si existiera una imputación, excluye tanto el retorno cuyo precio final fue imputado como el retorno siguiente cuyo precio inicial fue imputado; el número de filas afectadas queda en `returns_metadata.json`. También reporta tasas y rachas de retornos cero por IS/OOS sin excluir activos automáticamente.

## 4. Separar muestras rolling y no solapadas

**Problema:** mezclar ventanas H=5 solapadas con evaluación independiente infla el tamaño aparente.

**Implementación:** la estimación conserva rolling; la evaluación usa archivos `*_nonoverlap.csv`. Los splits se construyen antes de componer H, evitando ventanas cruzadas.

```powershell
.\.venv\Scripts\python.exe run.py --step transform
Get-Content .\outputs\returns_metadata.json
```

## 5. Introducir benchmarks comparables

**Problema:** MM se calibraba a retornos H=5 EWMA y Wiener a retornos diarios uniformes.

**Implementación:** Gaussian terminal, Student-t terminal e histórico EWMA reciben los mismos targets H. El alias Wiener solo se conserva por compatibilidad.

```powershell
.\.venv\Scripts\python.exe run.py --step benchmarks
.\.venv\Scripts\python.exe -m unittest tests.test_benchmarks -v
```

## 6. Evaluar distribuciones, no solo momentos

**Problema:** un buen ajuste in-sample no prueba calidad predictiva.

**Implementación:** CRPS, Energy Score, Variogram Score, correlación OOS, VaR, ES, Kupiec y Christoffersen sobre ventanas no solapadas.

```powershell
.\.venv\Scripts\python.exe run.py --step evaluate
Import-Csv .\outputs\tables\probabilistic_scores_summary.csv | Format-Table
```

Menor score es mejor; una tasa VaR cercana a 5% sin rechazo es deseable, pero una muestra corta implica baja potencia.

## 7. Cuantificar incertidumbre de los scores

**Problema:** ordenar promedios con diferencias de pocas millonésimas no indica si la ventaja es distinguible del ruido temporal. Además, ejecutar muchos contrastes sin corrección aumenta falsos positivos.

**Implementación:** `src/evaluation/scoring.py` guarda CRPS, Energy y Variogram por cada una de las 121 ventanas. Energy estima una sola vez por modelo su término entre escenarios con 100.000 pares, evitando un sorteo diferente por fecha. `src/evaluation/comparison.py` calcula `MM − benchmark`, remuestrea bloques móviles de cuatro ventanas durante 5.000 iteraciones, construye un IC95 básico con la distribución centrada bajo H0 y aplica Holm a la familia de nueve contrastes.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_comparison tests.test_scoring -v
.\.venv\Scripts\python.exe run.py --step evaluate
Import-Csv .\outputs\tables\probabilistic_score_differences.csv | Format-Table
```

Un valor negativo favorece a MM. El contraste registrado como primario es CRPS de MM contra Gaussiano. El periodo sigue siendo validación de desarrollo: un `p_Holm` bajo describe esta muestra, no convierte el resultado en confirmatorio.

## 8. Robustecer decisiones de portafolio

**Problema:** Markowitz/MaxSharpe pueden ser inestables y la cola ponderada se truncaba sin masa fraccionaria.

**Implementación:** ES exacto discreto, regularización L2, LP de CVaR, Equal Weight, inverse variance y HRP.

```powershell
.\.venv\Scripts\python.exe run.py --step portfolio
.\.venv\Scripts\python.exe -m unittest tests.test_portfolio -v
```

## 9. Backtest con fricción y walk-forward

**Problema:** faltaban deriva de pesos, turnover, costos e incertidumbre de la diferencia frente a un benchmark ingenuo.

**Implementación:** simulación diaria; evaluación estática OOS para MM y modelos generativos; rebalanceo trimestral walk-forward de EW/IV/HRP; costos en bps; y moving-block bootstrap del diferencial de Sharpe anualizado.

```powershell
.\.venv\Scripts\python.exe run.py --step backtest
Import-Csv .\outputs\tables\backtest_metrics.csv | Format-Table
Import-Csv .\outputs\tables\bootstrap_sharpe_differences.csv | Format-Table
```

Una estrategia no supera a Equal Weight de forma fuerte si el IC95 contiene cero.

La columna `evaluation_design` diferencia `static_single_shot_oos` de `expanding_window_walk_forward`; no deben presentarse como el mismo experimento.

## 10. Robustez frente a iliquidez sin look-ahead

**Problema:** una tasa alta de retornos cero puede reflejar negociación no sincrónica, precios estancados o costos de transacción no observados. Excluir activos después de mirar OOS contaminaría la validación.

**Implementación:** `research/liquidity_robustness.yaml` congela una regla que usa únicamente `zero_return_rate_is <= 3%` y `max_zero_run_is <= 5`. `src/analysis/liquidity_robustness.py` selecciona el universo, recalibra MM y los tres benchmarks con idénticos hiperparámetros, vuelve a optimizar portafolios y repite scores/backtest. OOS se adjunta solo como diagnóstico ex post; no participa en `selected`.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_liquidity_robustness -v
.\.venv\Scripts\python.exe run.py --step liquidity
Import-Csv .\outputs\robustness\liquidity\universe_selection.csv | Format-Table
Import-Csv .\outputs\robustness\liquidity\probabilistic_rank_comparison.csv | Format-Table
Import-Csv .\outputs\robustness\liquidity\portfolio_comparison_full_vs_liquid.csv | Format-Table
```

La prueba automatizada cambia de forma extrema todas las métricas OOS y exige que la selección no cambie. Los niveles de CRPS entre 15 y 11 activos no se comparan como si fueran el mismo problema; se estudia el ranking dentro de cada universo y la estabilidad económica.

## 11. Reproducibilidad y gobernanza

**Problema:** no había tests, lock, hashes ni fallo duro.

**Implementación:** suite `unittest`, `requirements-lock.txt`, `pyproject.toml`, manifest con SHA-256/versiones y `verify.py` con exit code no cero.

Cada etapa escribe además `outputs/lineage/<etapa>.json`. El manifiesto registra hashes de entradas, código relevante y salidas. Una etapa posterior se niega a consumir resultados si cualquiera de esos hashes cambió; así una nueva calibración MM no puede combinarse silenciosamente con scores o portafolios anteriores.

```powershell
.\.venv\Scripts\python.exe run.py --step snapshot
.\.venv\Scripts\python.exe verify.py --scope full
if ($LASTEXITCODE -ne 0) { throw "La verificacion fallo" }
```

El alcance `full` exige ocho etapas de linaje vigentes, incluida `liquidity_robustness`, y un snapshot que declare exactamente esa cadena.

## 12. Migrar de tesis a investigación independiente

El documento principal es `research/MM_Research_Report.tex`; no contiene profesor, universidad ni afirmaciones heredadas. El periodo ya observado se etiqueta validación de desarrollo. Los documentos y resultados previos permanecen como legacy para trazabilidad y no deben mezclarse con una nueva corrida.

## 13. Validar estabilidad con rolling-origin

**Problema:** un único corte OOS puede reflejar un régimen particular y producir rankings frágiles.

**Implementación:** `research/rolling_origin.yaml` congela cuatro folds expansivos. `src/analysis/rolling_origin.py` recalibra MM, Gaussiano, Student-t e histórico en cada origen; `src/evaluation/comparison.py` remuestrea bloques sin cruzar folds. La salida pooled pondera cada ventana y `model_stability_summary.csv` conserva rankings por fold.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_rolling_origin tests.test_comparison -v
.\.venv\Scripts\python.exe run.py --step rolling
Import-Csv .\outputs\robustness\rolling_origin\probabilistic_scores_by_fold.csv | Format-Table
Import-Csv .\outputs\robustness\rolling_origin\probabilistic_score_differences_pooled.csv | Format-Table
```

La etapa completa tarda aproximadamente dos minutos en la máquina de desarrollo. Para cambios puramente documentales se usa `scripts/release_check.ps1`; `-Full` exige además todos los artefactos y linajes de investigación.
