# Publicación para LinkedIn

Agregué a mi investigación el benchmark que cualquier revisor habría exigido
primero. Mi modelo perdió contra él en las tres métricas.

La pregunta del proyecto es sencilla y molesta:

**¿Ajustar casi exactamente los primeros cuatro momentos y la covarianza produce
mejores escenarios financieros fuera de muestra?**

Implementé MM-BCD —generación de escenarios discretos por ajuste de momentos vía
descenso en bloques— sobre 15 acciones chilenas, con validación rolling-origin de
cuatro folds y 169 ventanas no solapadas de cinco días.

Durante meses lo comparé contra tres controles: Gaussiano, Student-t e histórico
EWMA. Todos comparten un defecto que tardé en ver: **son distribuciones
estáticas**. Ninguno modela cómo evoluciona la volatilidad ni la dependencia
dentro del horizonte. Ganarle a un gaussiano estático es un listón bajo, y yo ni
siquiera lo estaba superando.

Así que implementé DCC-GARCH (Engle, 2002), el estándar de la literatura de
pronóstico multivariado financiero. Dos etapas: GARCH(1,1) por activo con
variance targeting, y correlación condicional dinámica sobre los residuos
estandarizados.

El resultado, en validación rolling-origin:

- DCC-GARCH obtiene el mejor CRPS, Energy Score y Variogram Score.
- Es el único modelo dentro del Model Confidence Set al 95% en dos de las tres.
- MM-BCD queda fuera en las tres, con los contrastes significativos tras Holm.

El diagnóstico de calibración explica el mecanismo. MM-BCD alcanza la **mejor
razón de dispersión de los cinco modelos** (0.995 contra un ideal de 1.000) y a
la vez el histograma PIT menos uniforme. Ajustar cuatro momentos no equivale a
ajustar una distribución, y las reglas de scoring propias evalúan la forma
completa. Es más: DCC-GARCH es el modelo **más sobredisperso** de todos y gana
igual, porque lo que decide no es la dispersión sino capturar la dependencia
condicional.

Hubo un hallazgo que no buscaba y que quizá sea el más útil. Bajo un ajuste único
proyectado 2.5 años, DCC-GARCH queda **último** en CRPS. Recalibrado en cada
origen, queda **primero**. El valor de un modelo condicional está en condicionar
al estado actual: congelarlo lo vuelve peor que una distribución estática. La
conclusión sobre un modelo dinámico depende del protocolo de recalibración tanto
como del modelo.

Antes de esto ya había corregido algo incómodo: los grados de libertad de mi
control Student-t estaban fijados a mano en 6.0, idénticos para quince activos
con curtosis muy distinta. Al estimarlos por verosimilitud en cada origen, el
rango real resultó ser 12 a 25, y tres conclusiones que yo mismo había publicado
dejaron de sostenerse.

Ambos episodios enseñan lo mismo: **una comparación solo es válida si el rival
está tan bien especificado como el candidato**. Es fácil ganarle a un benchmark
mal calibrado, y es igual de fácil perder contra uno inflado por accidente. En
los dos casos el número no significa nada.

La versión 0.7 incluye grados de libertad estimados en cada origen, ancho de
bloque del bootstrap por Politis-White, contracción de covarianza por
Ledoit-Wolf, Model Confidence Set, Diebold-Mariano con varianza HAC como ruta de
inferencia independiente, diagnósticos PIT con soporte igualado entre modelos,
178 tests, CI multiversión, nueve etapas de linaje y snapshots SHA-256.

Mi modelo no ganó. Pero ahora sé exactamente por qué, y el resultado es
defendible.

¿Qué otro control someterías a esta comparación?

https://github.com/tav0-m/mm-ipsa-research

#DataScience #QuantitativeFinance #TimeSeries #Python #OpenScience

Este proyecto es investigación independiente y no constituye asesoría de
inversión.
