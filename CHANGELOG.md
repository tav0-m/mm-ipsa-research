# Changelog

Todos los cambios relevantes de este proyecto se documentarán en este archivo.

## [0.12.0] - 2026-08-27

Auditoria de integridad estructural de la serie de precios. Ningun resultado
publicado cambia: la serie actual pasa la auditoria sin hallazgos bloqueantes.

### Anadido

- `data/integrity.py`. Los controles de calidad existentes miden cobertura, y un
  ajuste de evento corporativo mal aplicado tiene cobertura perfecta: produce un
  salto de nivel espurio que ningun control de faltantes puede ver. El modulo
  detecta ese patron por consistencia interna, sin requerir una segunda fuente.
- Discriminante por persistencia de nivel. Un desplome de mercado y un split no
  ajustado producen retornos igual de grandes; lo que los separa es que el
  segundo desplaza el nivel de forma permanente. Cada candidato se evalua
  comparando el nivel mediano antes y despues del salto.
- Informe de precios estancados. `AGUAS-A` pasa el 5,0% de las jornadas sin
  variacion de precio, con rachas de hasta seis dias, y la imputacion es cero:
  es negociacion real que no se movio. Sesga la volatilidad a la baja.
- Comando `mm-ipsa audit-data`, fuera del pipeline sellado.
- 19 pruebas nuevas, incluida la deteccion de un split inyectado en 30 de 30
  realizaciones independientes.

### Corregido

- La banda de deteccion de splits se media como porcentaje fijo. Un split real
  nunca produce la razon exacta porque el precio observado incorpora tambien el
  movimiento de mercado de esa jornada: con una tolerancia del 2% sobre una
  volatilidad diaria del 1,5%, uno de cada cinco splits reales pasaba sin
  marcarse. Ahora se expresa en escalas robustas del propio activo.

### Pendiente declarado

- El sello de preregistro cubre quince archivos de implementacion y ninguno esta
  en `src/mm_ipsa/data/`, donde `transform.py` construye las ventanas terminales
  y `download.py` aplica la imputacion. Un cambio en cualquiera alteraria los
  resultados sin figurar como drift. Ampliar el alcance del sello es una decision
  sobre el compromiso ya firmado y se deja explicita en vez de aplicarse.
- La reconciliacion contra un segundo proveedor sigue pendiente: no se localizo
  una fuente independiente y accesible que cubra este universo.

## [0.11.1] - 2026-08-26

Pase de calidad de codigo. Ningun resultado cambia: los scores agregados
coinciden con los publicados hasta el redondeo de la sexta cifra.

### Cambiado

- `objective.py` reescrito con nombres descriptivos. Se elimino `compute_errors`,
  un metodo de 35 lineas sin ningun consumidor en el proyecto, que ademas
  contenia el unico diccionario con claves en espanol y un bucle O(n^2) sobre
  pares de activos.
- `diagnostics.py`: `mh`/`mm` pasan a `historical`/`fitted`. La segunda era
  especialmente confusa porque `mm` es tambien el nombre del paquete.
  `mu_h`, `sig_h`, `sk_h`, `ku_h` y sus contrapartes reciben nombres completos.
- `bcd.py`: se condensan los comentarios que narraban correcciones historicas y
  se eliminan los separadores decorativos.

### Eliminado

- Los 22 separadores decorativos del arbol de codigo.
- Comentarios que repetian la linea siguiente en vez de explicarla.

Comentarios totales de 220 a 144. Densidad en `objective.py` de 22,5% a 2,9%; en
`diagnostics.py` de 6,6% a 2,4%.

## [0.11.0] - 2026-08-26

Congelamiento verificable para el test confirmatorio. El protocolo lo declaraba
desde v0.5.0 sin hacerlo efectivo.

### Anadido

- **Sello de preregistro** en `research/preregistration.yaml`, con el comando
  `mm-ipsa freeze`. Fija la fecha de inicio confirmatorio, la muestra minima, el
  contraste primario, la regla de decision y la lista de lo que queda prohibido
  tras el sello.
- El sello separa **especificacion** de **implementacion**. La primera
  -configuracion, protocolo y cortes temporales- no puede cambiar: la
  verificacion compara hashes y falla si alguno difiere. La segunda si puede,
  porque una reescritura numericamente equivalente no altera el experimento,
  pero cada archivo modificado queda enumerado.
- Contrato de verificacion que exige especificacion intacta y coherencia entre
  el estado declarado y la muestra confirmatoria disponible.
- Diecisiete pruebas del modulo, incluida la comprobacion de que un cambio en la
  configuracion rompe el sello y de que un cambio en el codigo no lo rompe.

### Decidido

- **Inicio confirmatorio: 2026-09-01**, con un minimo de 40 ventanas H=5 no
  solapadas antes de admitir cualquier lectura. Son unos diez meses de mercado.
- **El contraste primario pasa de MM contra Gaussiano a MM contra DCC-GARCH.**
  Es el estandar de la literatura y el unico dentro del Model Confidence Set. El
  cambio se declara antes de que exista dato confirmatorio, que es la unica
  circunstancia en que redefinir un contraste primario es legitimo.

## [0.10.0] - 2026-08-25

Ablacion controlada del efecto de recalibrar. Retracta una hipotesis que las
versiones anteriores conservaban como exploratoria.

### Anadido

- **Ablacion frozen contra refit sobre ventanas identicas.** Ambas variantes se
  puntuan en exactamente las mismas 169 ventanas, con el mismo universo y las
  mismas semillas, de modo que la muestra y el calendario dejan de confundirse
  con la actualizacion de los modelos. En el primer fold ambas coinciden por
  construccion y el experimento lo verifica.
- Once pruebas del modulo, incluida la comprobacion del montaje.

### Resultado

- **Recalibrar aporta a los cinco modelos en las tres reglas**: quince de quince
  contrastes significativos tras Holm, con todos los intervalos sobre cero.
- **El ranking no se invierte.** DCC-GARCH gana bajo calibracion congelada y bajo
  recalibracion por fold. El cambio de posicion que se observaba entre el split
  unico y el rolling-origin provenia de que ambos disenos evaluan periodos
  distintos, no de la cadencia de reajuste.
- DCC-GARCH se degrada dos a tres veces menos al congelarlo: 7.2 por ciento en
  Variogram Score frente a 19 a 22 por ciento de los cuatro modelos estaticos. Un
  modelo condicional arrastra estado que sustituye parcialmente a la
  recalibracion externa, y ese es el mecanismo real detras de su ventaja.

### Corregido

- Se retira de README y RESULTS la lectura de que el protocolo de recalibracion
  determinaba el veredicto. La ablacion la descarta.

El paso rolling pasa de 74 s a 176 s; el pipeline completo queda en 509 s.

## [0.9.0] - 2026-08-25

Cierra la limitacion de diseno que las versiones anteriores declaraban
pendiente. El experimento fue posible gracias a la optimizacion de v0.8.0.

### Corregido

- **Todos los generadores de escenarios se recalibran en cada fecha de
  rebalanceo** y de cada uno se derivan las tres carteras del protocolo. Hasta
  ahora los portafolios derivados de modelos se calibraban una vez y quedaban
  congelados 2,5 anos, mientras los baselines ingenuos se reajustaban cada
  trimestre. La v0.8.0 corrigio la comparacion emparejando disenos; esta corrige
  el diseno mismo.
- H4 pasa a contrastarse bajo un protocolo simetrico. **Ninguna de las
  diecisiete estrategias supera al Equal Weight tras la correccion de Holm.**

### Resultado

- El unico contraste que sobrevive sin ajustar es la minima varianza derivada de
  DCC-GARCH, con p de 0.0085; con Holm sobre diecisiete comparaciones queda en
  0.144.
- Las cuatro carteras de minima varianza derivadas de modelos son practicamente
  indistinguibles entre si, porque todas apuntan a la misma covarianza objetivo.
- Las variantes de minimo CVaR y maximo Sharpe rotan entre 3,2 y 5,3 veces el
  patrimonio frente a 0,35 del Equal Weight, con intervalos de hasta mas menos
  0,6 en Sharpe.

### Anadido

- Modulo de calendarios walk-forward por modelo, con tabla de trazabilidad que
  registra filas de entrenamiento, ultimo dato usado y soporte de cada modelo en
  cada origen.
- Nueve pruebas del modulo, incluida la verificacion de que el entrenamiento
  nunca alcanza la fecha de decision y de que la ventana crece de forma
  monotona.

El paso de backtest pasa de 14 s a 243 s por las diez recalibraciones; el
pipeline completo queda en 401 s.

## [0.8.0] - 2026-08-25

Version centrada en eficiencia. El trabajo destapo un defecto metodologico y su
correccion. Los datos crudos no cambian y los cuatro controles quedan identicos.

### Corregido

- **La solucion de MM se publica como mezcla de los starts elegibles**, no como
  el de menor G. El multi-start ya calculaba varias soluciones y descartaba
  todas menos una; ese descarte era el paso fragil, porque el objetivo es no
  convexo y la eleccion se movia con diferencias numericas irrelevantes. Medido
  con ocho semillas, esa variacion era del mismo orden que varios de los efectos
  contrastados. La mezcla no cuesta computo adicional, reduce la sensibilidad a
  la semilla 3.1x en CRPS y 2.4x en Energy, y mejora los tres scores de MM.
- El contrato de estacionariedad pasa a exigirse **por miembro del ensemble** en
  vez de sobre la solucion publicada. Es mas estricto: antes bastaba un punto
  estacionario, ahora deben serlo todos.
- MMObjective valida consistencia entre x y p en lugar de un tamano de soporte
  fijo, y la referencia uniforme de la KL usa el soporte efectivo.

### Rendimiento

Tres reescrituras, verificadas como numericamente equivalentes:

- Segunda etapa DCC como filtro IIR de primer orden resuelto en C, con Cholesky
  y formas cuadraticas por lote: **6.2x**.
- Indices del bootstrap por bloques generados por difusion en vez de una lista
  por replica: **13.7x** en los doce contrastes, **8.6x** en el Model Confidence
  Set.
- Potencias de las desviaciones por multiplicacion encadenada en lugar de
  np.power: **10.2x** en la calibracion MM.

Suite de pruebas de 69 s a 7 s; pipeline completo de 263 s a 173 s.

### Anadido

- Test de regresion que contrasta la verosimilitud DCC vectorizada contra una
  implementacion ingenua escrita directamente desde la formula.
- Diagnostico de publicacion con tamano del ensemble, dispersion del objetivo
  entre miembros y residuos de estacionariedad por miembro.

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
