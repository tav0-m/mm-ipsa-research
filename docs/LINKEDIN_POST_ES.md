# Publicación para LinkedIn

**MM-IPSA Research** es una investigación cuantitativa independiente sobre
generación de escenarios probabilísticos para quince acciones chilenas.

La pregunta que intenta responder es concreta: **¿reproducir con precisión la
media, la covarianza, la asimetría y la curtosis produce mejores pronósticos
fuera de muestra, y mejores decisiones de portafolio?**

Mi modelo, MM-BCD, genera escenarios discretos ajustando esos cuatro momentos por
descenso en bloques. Lo consigue con un error relativo de 0,00%. La investigación
existe para averiguar si esa precisión sirve de algo.

## Qué se construyó para responderla

El objetivo no era ganar una comparación, sino montar una capaz de detectar
cuándo y por qué el modelo pierde:

- Cuatro controles, incluido **DCC-GARCH**, el estándar de la literatura de
  pronóstico multivariado.
- Reglas de scoring propias —CRPS, Energy Score, Variogram Score— sobre 169
  ventanas no solapadas y cuatro folds rolling-origin.
- Inferencia por dos rutas independientes: bootstrap por bloques con ancho
  elegido automáticamente, y Diebold-Mariano con varianza HAC.
- **Model Confidence Set**, que responde qué modelos no pueden descartarse como
  óptimos en vez de acumular comparaciones de a pares.
- Linaje con hashes que impide publicar resultados obsoletos en silencio.

## Qué encontró

**DCC-GARCH obtiene el mejor valor en las tres reglas** y es el único dentro del
Model Confidence Set en Energy y Variogram Score. MM-BCD queda fuera en las tres.

El cuadro es más matizado de lo que suele contarse. En CRPS **ninguna diferencia
entre modelos resulta distinguible**, ni siquiera frente a DCC-GARCH. MM sí
supera de forma significativa al Gaussiano y al histórico en dependencia entre
activos, y pierde frente al control dinámico en Energy y Variogram.

En decisiones de portafolio, con los cinco generadores recalibrados en la misma
cadencia que los baselines, **ninguna de diecisiete estrategias supera al Equal
Weight** tras corregir por multiplicidad.

El diagnóstico de calibración explica el mecanismo: MM-BCD queda más cerca de la
dispersión ideal que cualquier otro modelo, pero no logra el mejor histograma
PIT. Ajustar la escala no equivale a ajustar la distribución completa, y menos su
dependencia condicional.

## Lo que cambió por el camino

Tres correcciones invalidaron conclusiones que yo mismo había publicado:

Los grados de libertad de mi control Student-t estaban fijados a mano en 6,0.
Estimarlos por verosimilitud dio un rango real de 12 a 25, y tumbó tres
resultados anteriores.

La solución de MM se publicaba eligiendo el mejor de varios arranques del solver.
Como el objetivo no es convexo, esa elección se movía con diferencias numéricas
irrelevantes, y esa variación resultó ser **del mismo orden que los efectos que
estaba midiendo**. Ahora se publica la mezcla de los arranques válidos.

Y observé que DCC-GARCH cambiaba de último a primero según el diseño de
evaluación. Parecía indicar que el protocolo decidía el veredicto. Una ablación
controlada sobre exactamente las mismas ventanas lo descartó: el ranking no se
invierte, el cambio venía de que ambos diseños evalúan periodos distintos.

## Hacia dónde va

El periodo evaluado ya fue observado, de modo que nada de esto puede llamarse
confirmatorio. Por eso el proyecto está ahora **congelado**: la especificación se
selló el 26 de agosto, la evaluación confirmatoria empieza el 1 de septiembre y
exige al menos cuarenta ventanas antes de admitir lectura alguna. Son unos diez
meses de mercado.

El sello se verifica por hash. Si la configuración, el protocolo o los cortes
temporales cambian, `mm-ipsa verify` falla. Cualquiera puede prometer que no
reajustará nada cuando lleguen los datos nuevos; el punto es que el código lo
detecte.

El proyecto tiene 225 pruebas automatizadas, verificación de tipos e integridad
en cada etapa, y todo el código abierto.

Mi modelo no ganó. Pero sé exactamente por qué, y eso sí es un resultado.

https://github.com/tav0-m/mm-ipsa-research

#DataScience #QuantitativeFinance #TimeSeries #Python #OpenScience

Investigación independiente. No constituye asesoría de inversión.
