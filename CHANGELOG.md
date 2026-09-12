# Changelog

Todos los cambios relevantes de este proyecto se documentarán en este archivo.

## [0.18.0] - 2026-09-11

Ampliacion exploratoria del universo y medicion de como escala MM-BCD. El sello
prohibe modificar el universo de activos, de modo que la ampliacion vive en una
configuracion aparte y `config.yaml` no se toca: la especificacion sellada sigue
intacta y el test confirmatorio no se ve afectado.

### Anadido

- `config_extended.yaml` con veintinueve activos. Los catorce anadidos se
  seleccionaron con criterios fijados antes de mirar resultados y derivados del
  propio universo incumbente, no elegidos para que convengan: integridad sin
  hallazgos bloqueantes, jornadas sin variacion por debajo del 4,99% del peor
  incumbente, y racha maxima por debajo de sus seis jornadas.
- Descuento del movimiento comun en la deteccion de eventos corporativos. Un
  split altera una sola serie y un desplome altera todas, pero ambos persisten
  como desplazamiento de nivel, de modo que el filtro anterior no los separaba.
- 3 pruebas nuevas.

### Corregido

- La caida de VAPORES del 18 de marzo de 2020, del 32,7%, quedaba marcada como
  posible evento corporativo. Era un falso positivo: el mercado completo cayo
  14,1% esa jornada, CAP cayo 27,5% dentro del universo ya admitido, y el precio
  siguio evolucionando despues. El descuento transversal la limpia y conserva
  los seis eventos de LTM, que si son idiosincraticos.
- El descuento se aplica solo con al menos cinco columnas. Con tres, la mediana
  transversal es practicamente una observacion y el propio salto la arrastra: la
  deteccion de un split inyectado cae de treinta sobre treinta a veintisiete.

### Seleccion

De veintidos candidatos con cobertura completa quedaron catorce. Siete se
rechazaron por iliquidez y el corte resulto natural: el peor admitido registra
3,80% de jornadas sin variacion y el menos malo de los rechazados 6,61%, tras lo
cual el grupo salta al rango de 25 a 41%. ANDINA-A alcanza 40,8% y HITES encadena
veintiocho jornadas sin moverse. LTM se rechazo por integridad pese a superar el
filtro de liquidez.

### Resultado: el metodo escala, la dependencia no

Con los mismos quinientos escenarios, pasar de quince a veintinueve activos
multiplica los targets por 3,16 y el tiempo por 3,71.

| Panel | n | Covarianzas | Targets | F total | Segundos |
|---|---:|---:|---:|---:|---:|
| Actual | 15 | 105 | 165 | 9,22e-11 | 9,8 |
| Extendido | 29 | 406 | 522 | 5,23e-10 | 36,3 |

El error relativo maximo distingue dos comportamientos opuestos:

| Panel | Media | Varianza | Tercer momento | Cuarto momento | Covarianza |
|---|---:|---:|---:|---:|---:|
| 15 activos | 7,8e-06 | 1,1e-06 | 7,6e-06 | 2,3e-07 | 4,6e-05 |
| 29 activos | 1,2e-05 | 1,4e-06 | 4,1e-06 | 1,4e-07 | **2,4e-03** |

**Los momentos marginales no se degradan**; alguno incluso mejora. **La
covarianza se degrada cincuenta y dos veces.** La asimetria es estructural: los
marginales son cuatro por activo y crecen de forma lineal, mientras las
covarianzas crecen como el cuadrado, y el soporte de quinientos escenarios tiene
capacidad fija para representar dependencia.

El hallazgo no es aislado. La debilidad de MM-BCD medida en las versiones
anteriores estaba siempre en dependencia -Energy y Variogram Score, atribucion de
cola- y ampliar el universo la agrava en la direccion esperada. Un uso serio del
universo ampliado exigiria elevar `N_scenarios`, y medir cuanto, es la pregunta
natural siguiente.

## [0.17.1] - 2026-09-11

Consolidacion de resultados. Ningun calculo cambia.

### Anadido

- `research/RESULTS_20260911.md`. Los cinco diagnosticos de riesgo vivian solo
  en entradas del changelog, dispersos entre cinco versiones. El informe los
  reune con la convencion del proyecto y explicita en cada uno que sobrevive a
  la correccion por multiplicidad y que no.
- Seccion de comportamiento en la cola en el README, que mostraba como resultado
  principal unicamente la tabla de scoring de la version 0.10.0.

### Cambiado

- La descripcion del proyecto declara ahora las dos vias de evaluacion. La
  predictiva juzga toda la distribucion; la de riesgo juzga la cola, donde un
  buen score promedio puede esconder un mal comportamiento.
- `RESULTS_20260825.md` pasa a citarse como resultados predictivos y no como
  resultados actuales sin mas, porque ya no es el unico informe vigente.

## [0.17.0] - 2026-09-11

Cierre del bloque de risk management: la calibracion estresada, que es la
respuesta que el marco regulatorio da al problema diagnosticado en la version
anterior. Se contrasta en vez de asumirse.

### Anadido

- `analysis/stressed_calibration.py`. Calibra los cinco generadores dos veces
  sobre la misma especificacion, con el tramo tensionado de la muestra y con el
  tramo en calma, y evalua ambas sobre el mismo periodo posterior. La unica
  diferencia entre las dos es la historia que las alimenta.
- Medicion de las dos caras del intercambio. Un informe que solo muestre la
  reduccion de excesos presenta media pregunta: el limite estresado es mas alto
  en todo momento, tambien en calma, y ese conservadurismo es capital
  inmovilizado.
- 12 pruebas nuevas.

### Resultado

Los tramos de calibracion difieren casi al doble en volatilidad anualizada:
16,6% en 2022-2023 contra 31,6% en 2020-2021.

| Calibracion | ES | Capital | Excesos en calma | Excesos en tension |
|---|---:|---:|---:|---:|
| Calma | -0,033 a -0,038 | 1,00 | 1,2 - 7,4% | 14,8 - 25,9% |
| Estres | -0,082 a -0,096 | 2,15 - 2,91 | 0,0% | 0,0% |

La calibracion estresada elimina los excesos en los dos regimenes y en los cinco
modelos, y cobra entre 2,15 y 2,91 veces el capital. El remedio funciona y no es
gratis.

El capital exigido ordena los modelos. MM-BCD necesita 2,91 veces su limite en
calma para alcanzar adecuacion y DCC-GARCH solo 2,15, un veintiseis por ciento
menos. La razon esta en el punto de partida: la calibracion en calma de MM-BCD
produce la cola mas fina de las cinco, -0,033 frente a -0,038 del historico, de
modo que es la que mas debe inflarse. Coincide con las tres versiones previas.

### Limites

Cero excesos en ciento ocho ventanas es una sobrecorreccion, no un ajuste fino:
con el nivel nominal del cinco por ciento se esperarian unos cinco. La
comparacion entre modelos bajo calibracion estresada mide por eso cuanto capital
pide cada uno, y no cual acierta mejor, porque en adecuacion todos saturan.

El tramo etiquetado como calma para calibrar, 2022-2023, no es la calibracion
publicada del proyecto, que pondera 2020-2023 con decaimiento exponencial. Las
cifras de esta tabla no son comparables con las de la version anterior.

## [0.16.0] - 2026-09-11

Cuarto bloque de risk management: prociclicidad. Un modelo puede estar bien
calibrado en promedio y fallar justo cuando importa, que es la critica central a
los modelos internos y la razon de que el marco regulatorio exija una
calibracion sobre periodo de estres.

### Anadido

- `evaluation/procyclicality.py`. El diagnostico no construye un estres
  sintetico: lo busca en los datos. El periodo de evaluacion contiene un cambio
  de regimen al alza, con volatilidad anualizada que pasa de 11,1% en el segundo
  semestre de 2024 a 20,6% en el primero de 2026, y las dos peores ventanas de
  veintiuna jornadas ocurren ambas en 2026.
- Clasificacion estrictamente ex-ante. Un retorno terminal fechado en ``t`` cubre
  ``[t - H + 1, t]``, de modo que la volatilidad que lo clasifica se mide sobre
  jornadas anteriores a ``t - H + 1``. Dos pruebas lo fijan: un shock dentro de
  la ventana no altera su propia etiqueta, y uno anterior si la altera.
- Prueba exacta de Fisher sobre la concentracion de excesos, que evita la
  aproximacion asintotica con pocos eventos.
- 15 pruebas nuevas.

### Resultado

Los excesos se concentran en el regimen tensionado en las quince combinaciones
de modelo y cartera, sin excepcion. Las tasas de exceso en calma van de 1,2% a
8,6%; en tension, de 14,8% a 29,6%.

| Modelo | Calma | Tension |
|---|---:|---:|
| DCC-GARCH | 1,2 - 2,5% | 14,8% |
| Historico EWMA | 1,2 - 3,7% | 18,5 - 22,2% |
| Gaussiano | 2,5 - 7,4% | 22,2 - 25,9% |
| Student-t | 2,5 - 7,4% | 22,2 - 25,9% |
| MM-BCD | 4,9 - 8,6% | 22,2 - 29,6% |

DCC-GARCH mantiene la tasa mas baja bajo tension en sus tres carteras, tres
veces el nominal frente a casi seis de MM-BCD con MinCVaR. Arrastra estado
condicional, de modo que su limite reacciona al cambio de regimen mientras los
generadores estaticos siguen describiendo la calma que ya paso.

### Limites

Tras la correccion de Holm sobre las quince combinaciones sobreviven cuatro. Y
hay una restriccion mas seria que la multiplicidad: las veintisiete ventanas
tensionadas forman un solo episodio contiguo desde junio de 2025. La prueba de
Fisher las trata como ensayos independientes, lo que exagera la evidencia. El
diagnostico documenta lo que ocurrio en un cambio de regimen, y no establece una
propiedad general de los modelos.

## [0.15.0] - 2026-09-11

Tercer bloque de risk management: atribucion de cola. Los contrastes anteriores
respondian si el ES predicho acertaba en magnitud. Este responde si acierta en
composicion, que es la pregunta accionable para una funcion de riesgo porque
determina que se cubre.

### Anadido

- `evaluation/attribution.py`. El ES es homogeneo de grado uno en los pesos, de
  modo que el teorema de Euler lo descompone de forma exacta: la contribucion de
  cada posicion es su peso por la esperanza condicional a que la cartera este en
  cola. La aditividad no es automatica en soporte discreto y exige repartir la
  masa del cuantil igual que el ES agregado; una prueba comprueba que los
  componentes suman el total a precision de maquina y que ese total coincide con
  `lower_tail_mean`.
- Contraste entre composicion predicha y realizada, evaluado sobre las
  submuestras disjuntas del horizonte y con la cola ensanchada al veinte por
  ciento, porque al cinco quedan menos de diez observaciones realizadas.
- 16 pruebas nuevas.

### Corregido

- La primera version comparaba las contribuciones ya multiplicadas por el peso.
  Como el vector de pesos es identico en la descomposicion predicha y en la
  realizada, una cartera concentrada producia correlacion cercana a uno sin que
  el modelo acertara nada: MinCVaR y MaxSharpe, con n efectivo de 6 frente a 15
  de MinVariance, marcaban 0.99. La medida de acierto se calcula ahora sobre las
  esperanzas condicionales, que es lo unico que el modelo aporta. La version
  confundida se conserva como descriptor de la cartera, etiquetada como tal.

### Resultado

Con la metrica corregida ningun modelo informa bien la composicion de la cola:
las correlaciones caen de 0.99 a un rango de -0.15 a +0.52, con media +0.22.
Los modelos aciertan aproximadamente la magnitud de la perdida y son casi ciegos
a su origen.

El patron por tipo de cartera es sistematico. MinVariance y MaxSharpe dan
correlacion positiva en los cinco modelos, alrededor de +0.35. MinCVaR da
correlacion negativa en cuatro de cinco. El contraste pareado dentro de cada
modelo arroja una diferencia media de -0.40, negativa en 23 de 25 combinaciones
de modelo y submuestra.

Ese resultado es coherente con la severidad de cola de la version anterior y
sugiere el mismo mecanismo: MinCVaR elige pesos que minimizan la cola segun el
propio modelo, de modo que carga sobre los activos cuyo riesgo de cola el modelo
subestima. La optimizacion selecciona sobre el error del modelo.

La evidencia no esta establecida. Los 25 pares comparten modelo o submuestra y
no son independientes; tratando cada modelo como una observacion quedan cuatro
signos negativos de cinco, con valor p de 0.19. Con cinco modelos un test de
signos no puede bajar de 0.031 ni con unanimidad.

## [0.14.0] - 2026-09-11

Segundo bloque de risk management: la mirada del regulador. El contraste de ES
pasa de evaluar activos aislados a evaluar la cartera efectivamente mantenida, y
se anade la clasificacion del Comite de Basilea.

### Anadido

- `evaluation/regulatory.py`. El semaforo de Basilea -verde hasta cuatro
  excepciones, amarilla de cinco a nueve, roja diez o mas- esta definido para
  doscientas cincuenta jornadas al noventa y nueve por ciento. Esos umbrales
  salen de la binomial y trasladarlos a otra muestra cambia el error de tipo I
  sin avisar. Las zonas se derivan aqui de la probabilidad acumulada, de donde
  provienen, y una prueba comprueba que reproducen la tabla publicada, zonas y
  multiplicadores de capital, en la configuracion del Comite.
- Declaracion de adecuacion muestral. Con pocas excepciones esperadas el color
  del semaforo dice mas del tamano de la muestra que del modelo, y el informe lo
  advierte en vez de presentar una clasificacion sin potencia.
- `observations_for_power`. Hacen falta 905 jornadas, cerca de tres anos y medio,
  para clasificar fuera de la zona verde con potencia del ochenta por ciento un
  modelo que subestima el riesgo al doble. Esa debilidad del contraste de VaR al
  noventa y nueve por ciento es parte de por que el marco migro a ES.
- `portfolio_shortfall_table`. La predictiva de cada cartera es la proyeccion de
  los escenarios sobre sus pesos, derivados con los optimizadores del propio
  proyecto.
- 20 pruebas nuevas.

### Resultado

Severidad de cola por cartera, es decir cuanto peores fueron las perdidas
realizadas frente al ES predicho:

| Modelo | MaxSharpe | MinVariance | MinCVaR |
|---|---:|---:|---:|
| DCC-GARCH | **0.979** | 0.996 | 1.038 |
| Student-t | 1.016 | 1.123 | 1.181 |
| Gaussiano | 1.035 | 1.153 | 1.187 |
| Historico EWMA | 1.101 | 1.261 | 1.255 |
| MM-BCD | 1.123 | 1.188 | **1.309** |

DCC-GARCH es el unico modelo con carteras conservadoras. MM-BCD con MinCVaR
alcanza 1.309 y es el unico contraste que sobrevive a Holm sobre las quince
combinaciones, con un valor p ajustado de 0.0037: las perdidas de cola realizadas
superaron en casi un tercio al ES predicho.

El patron es sistematico. MaxSharpe resulta la cartera mejor calibrada en los
cinco modelos, y MinCVaR la peor en cuatro de cinco. La cartera construida
explicitamente para minimizar riesgo de cola es la que peor calibra su cola, y el
mecanismo es que MinCVaR optimiza contra el conjunto de escenarios del propio
modelo: si esos escenarios subestiman la cola, el optimizador concentra posicion
justo donde el modelo se equivoca mas. Es el error de estimacion amplificado por
la optimizacion, localizado en la cola.

El semaforo es indicativo y no concluyente: con 1.21 excepciones esperadas por
cartera la muestra no alcanza el minimo que el propio modulo exige, y asi se
reporta.

## [0.13.0] - 2026-09-11

Primer bloque de risk management. El pipeline calculaba Expected Shortfall desde
la version inicial y nunca lo contrastaba: para VaR existian Kupiec,
Christoffersen y el test conjunto, y para ES no habia nada. La asimetria importa
porque la revision del marco de riesgo de mercado de Basilea III sustituyo VaR
por ES como medida regulatoria, de modo que el proyecto validaba la medida que la
industria dejo atras.

### Anadido

- `evaluation/expected_shortfall.py`. ES no es elicitable y por eso no admite un
  backtest por conteo de excedencias como VaR. Se implementan los estadisticos Z1
  y Z2 de Acerbi y Szekely (2014), cuya distribucion nula se obtiene simulando
  desde la propia predictiva. Aqui esa simulacion es exacta porque la predictiva
  del proyecto ya es discreta.
- Veredicto conjunto sobre ambos estadisticos. Un estudio de potencia propio
  muestra que ninguno domina: frente a una predictiva un veinte por ciento
  estrecha, Z2 rechaza el 78% de las veces y Z1 el 15%; frente a una cola t(4) la
  relacion se invierte, 65% contra 5%. Z2 es ciego a la cola pesada porque esta
  produce excedencias menos frecuentes pero mas profundas y ambos efectos se
  compensan en el estadistico incondicional. Un veredicto sobre uno solo dejaba
  pasar justo ese modo de fallo.
- Submuestras disjuntas para horizonte compuesto. Los retornos terminales a cinco
  dias calculados cada jornada comparten cuatro dias con su vecino, con
  autocorrelacion de 0,77 en el primer rezago. Aplicar el contraste directamente
  daria valores p muy por debajo de los correctos. Se contrastan las cinco
  submuestras disjuntas y se reporta si el veredicto cambia segun el
  desplazamiento.
- `analysis/shortfall_report.py` y comando `mm-ipsa shortfall`.
- 23 pruebas nuevas, incluidas calibracion de tamano y potencia empiricas.

### Resultado

Ordenados por severidad de cola, es decir cuanto peores fueron las perdidas
realizadas frente al ES predicho:

| Modelo | ES predicho | ES realizado | Severidad |
|---|---:|---:|---:|
| DCC-GARCH | -0.0842 | -0.0815 | 0.971 |
| Student-t | -0.0722 | -0.0733 | 1.025 |
| Gaussiano | -0.0708 | -0.0745 | 1.062 |
| MM-BCD | -0.0662 | -0.0713 | 1.086 |
| Historico EWMA | -0.0652 | -0.0703 | 1.091 |

DCC-GARCH es el unico conservador. MM-BCD subestima la cola en un 8,6%, penultimo
del conjunto, lo que es coherente con el resultado central del proyecto y le
anade un mecanismo: ajustar los cuatro primeros momentos no impone ninguna
restriccion sobre la cola mas alla del cuantil del cinco por ciento.

Ninguna diferencia sobrevive a la correccion de Holm sobre los quince activos.
El orden de las estimaciones puntuales es consistente, la evidencia no alcanza
significancia con esta muestra, y se reporta asi.

### Nota sobre el sello

`evaluation/` esta dentro del alcance sellado, de modo que el modulo nuevo figura
como cambio de implementacion en la auditoria del preregistro. La especificacion
permanece intacta y el test confirmatorio no se ve afectado: el contraste
primario registrado sigue siendo CRPS de MM contra DCC-GARCH.

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
- Carrusel de cinco imágenes para comunicar resultados, incertidumbre y límites.
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
- Informe metodológico, ficha pública y activos de comunicación.

### Resultado principal

- Student-t obtuvo el menor CRPS pooled.
- MM-BCD no mostró superioridad predictiva general y se conserva como resultado negativo informativo.

### Estado

- Release de investigación en validación de desarrollo.
- No constituye asesoría de inversión ni un test futuro sellado.
