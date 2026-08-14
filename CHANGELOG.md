# Changelog

Todos los cambios relevantes de este proyecto se documentarán en este archivo.

## [0.7.0] - 2026-08-14

Incorpora el competidor que faltaba. Los datos crudos no cambian.

### Añadido

- **DCC-GARCH como cuarto control** (Engle, 2002), estimado en dos etapas sin
  dependencias nuevas: GARCH(1,1) por activo con variance targeting y
  cuasi-verosimilitud gaussiana, más correlación condicional dinámica sobre los
  residuos estandarizados. Las innovaciones de simulación son t multivariadas
  con grados de libertad estimados de los residuos.
- Ambas etapas se validan recuperando parámetros conocidos de datos simulados.
- El control se reestima en cada origen del rolling-origin y en el universo
  líquido, y su diagnóstico de ajuste se persiste por fold.

### Resultado

- **DCC-GARCH obtiene el mejor valor en las tres reglas de scoring** y gana tres
  de los cuatro folds en CRPS. Es el único modelo dentro del Model Confidence
  Set en Energy y Variogram Score.
- MM-BCD queda fuera del conjunto en las tres reglas, con los tres contrastes
  significativos tras Holm.
- La posición de DCC-GARCH depende del diseño de evaluación: último en CRPS bajo
  un ajuste único proyectado 2.5 años, primero al recalibrarse en cada origen.

### Corregido

- Los contratos de verificación derivan el número de contrastes de la
  configuración en lugar de fijarlo en nueve, que ataba el protocolo a tener
  exactamente tres controles.
- La portada de release deriva la versión y el número de pruebas en lugar de
  llevarlos escritos a mano, donde ya habían quedado obsoletos.
- `release_assets` tolera controles no catalogados en vez de fallar con
  `KeyError`.

## [0.6.0] - 2026-08-13

Release de rigor metodológico. Los datos crudos no cambian —la descarga se
revalida por hash sin consultar al proveedor—, de modo que toda diferencia en los
resultados proviene de correcciones de método.

### Corregido

- **Los grados de libertad del Student-t se estiman por verosimilitud de perfil**
  en lugar de imponerse como la constante `6.0`. Se reestiman en cada origen del
  rolling-origin; el rango obtenido es 12 a 25. Tres conclusiones de v0.5.0
  dejan de sostenerse: la ventaja de MM en Energy Score frente al histórico y sus
  desventajas en Variogram Score frente a Student-t e histórico.
- **El ancho de bloque del bootstrap lo elige Politis-White (2004)** con la
  corrección de Patton, Politis y White (2009), sobre autocovarianzas agrupadas
  dentro de folds, en lugar de la constante `4` justificada por analogía.
- **La contracción de covarianza usa la intensidad óptima de Ledoit-Wolf**
  (`0.0612` estimado) en lugar de la constante `0.10`, que contradecía el propio
  protocolo al no provenir de validación temporal.
- **Los portafolios se contrastan contra el baseline de su mismo diseño de
  evaluación.** Antes toda estrategia se comparaba contra `WF_EqualWeight`, de
  modo que las carteras estáticas competían contra un baseline recalibrado
  trimestralmente y H4 no era interpretable. El efecto del rebalanceo se reporta
  ahora como familia separada.
- Holm se aplica a los contrastes de portafolio, que antes no recibían ninguna
  corrección de multiplicidad.
- `_feasible_weights` verifica el tope por activo tras renormalizar, en lugar de
  devolver en silencio una cartera infactible.
- `simulate_strategy` falla si una fecha de rebalanceo no existe en el índice, en
  lugar de omitirla sin dejar rastro.
- `nearest_psd` usa un piso de autovalores relativo al mayor; el piso absoluto
  `1e-12` no regularizaba matrices con entradas de orden `1e-4`.
- Las banderas de calidad de datos se calculan por segmento; la tasa agregada
  ocultaba deterioros concentrados en la ventana de evaluación.
- `portfolio_diagnostics` resta la tasa libre de riesgo, igual que
  `maximum_sharpe`.
- Los diagnósticos reportan exceso de curtosis, de modo que el cero corresponde a
  colas gaussianas.
- `evaluate_scenarios_detailed` exige que los identificadores conserven el orden
  cronológico al pasar a texto, supuesto del que depende el bootstrap por bloques.

### Añadido

- Model Confidence Set de Hansen, Lunde y Nason (2011) en el análisis principal y
  en rolling-origin.
- Contraste de Diebold-Mariano con varianza HAC de Newey-West y corrección de
  Harvey, Leybourne y Newbold, como ruta de inferencia independiente del
  bootstrap. Coincide con el bootstrap en los nueve contrastes.
- Diagnósticos de calibración por transformada integral de probabilidad, con
  soporte igualado entre modelos. Bajo especificación correcta un ensemble de 500
  escenarios rechaza uniformidad cerca del 27% de las veces frente al 5% nominal,
  por lo que sin igualar se mediría resolución en lugar de calibración.
- Análisis de sensibilidad del ancho de bloque sobre una grilla configurable.
- Artefactos auditables `student_t_df_estimation.json`,
  `student_t_df_by_fold.csv`, `model_confidence_set.csv`,
  `calibration_pit_by_asset.csv` y `calibration_pit_summary.csv`, incorporados al
  linaje.
- Cobertura de pruebas de 49 a 159, incluidos módulos antes sin tests directos
  (`pipeline.py`, `verification.py`) y casos degenerados del solver BCD.

### Eliminado

- Bloque `wiener` de la configuración y alias `wiener_scenarios`, ambos sin
  consumidores.
- Dependencia `seaborn` del lockfile, no declarada ni importada.

## [0.5.0] - 2026-08-12

### Cambiado

- Migración al layout estándar `src/mm_ipsa` y eliminación del paquete genérico `src`.
- CLI única y multiplataforma mediante `mm-ipsa`.
- Ejecución reanudable basada en hashes con `--resume` y planificación sin efectos con `--plan`.
- CI ampliado a Python 3.11 y 3.12, Ruff, Pyright y construcción del paquete.
- Contratos de tipos explícitos para fechas, linaje, resultados rolling-origin y el solver BCD.
- Carrusel de cinco imágenes para comunicar resultados, incertidumbre y límites en LinkedIn.
- Limpieza de código muerto, alias históricos y documentación ajena a la release.

### Compatibilidad

- `run.py` y `verify.py` permanecen como envoltorios temporales para comandos anteriores.

## [0.4.0] - 2026-08-10

### Añadido

- Validación rolling-origin expansiva con cuatro folds y 169 ventanas OOS H=5.
- Recalibración por fold de MM-BCD, Gaussiano, Student-t e histórico EWMA.
- Inferencia pareada mediante moving-block bootstrap y corrección de Holm.
- Robustez de liquidez con selección exclusivamente in-sample.
- Nueve etapas de linaje, snapshots SHA-256 y 45 pruebas automatizadas.
- Informe metodológico, ficha pública y activos para comunicación en LinkedIn.

### Resultado principal

- Student-t obtuvo el menor CRPS pooled.
- MM-BCD no mostró superioridad predictiva general y se conserva como resultado negativo informativo.

### Estado

- Release de investigación en validación de desarrollo.
- No constituye asesoría de inversión ni un test futuro sellado.
