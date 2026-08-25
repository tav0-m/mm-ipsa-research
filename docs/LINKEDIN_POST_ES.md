# Publicación para LinkedIn

Estoy desarrollando **MM-IPSA Research**, una investigación independiente sobre
generación de escenarios probabilísticos para 15 acciones chilenas.

La pregunta central es directa: **¿reproducir casi exactamente la media, la
covarianza, la asimetría y la curtosis produce mejores pronósticos fuera de
muestra?**

Mi modelo, MM-BCD, genera 500 escenarios mediante ajuste de momentos y
optimización por descenso en bloques. Lo comparé con cuatro benchmarks:
Gaussiano, Student-t, histórico EWMA y DCC-GARCH.

La evaluación principal usa un protocolo rolling-origin expansivo, cuatro folds
y 169 ventanas no solapadas de cinco días. Cada modelo se recalibra al inicio de
cada fold usando únicamente información disponible hasta esa fecha.

## Qué muestran los cuatro gráficos

**1. Desempeño agregado.** DCC-GARCH obtiene la menor pérdida en CRPS, Energy
Score y Variogram Score. MM-BCD no transforma su ajuste casi exacto de momentos
en superioridad predictiva.

**2. Estabilidad temporal.** DCC-GARCH gana tres de los cuatro folds en CRPS,
pero queda quinto en 2024; Student-t gana ese período. El resultado agregado es
favorable, aunque no es uniforme en el tiempo.

**3. Incertidumbre estadística.** Frente a DCC-GARCH, la pérdida relativa de
MM-BCD es +0,75% en CRPS, +0,90% en Energy y +3,12% en Variogram. Los tres IC95%
quedan sobre cero y los contrastes sobreviven la corrección de Holm.

**4. Diagnóstico de calibración.** MM-BCD presenta el menor error de dispersión,
pero no el mejor índice de fiabilidad PIT. DCC-GARCH gana los scores aun con una
distribución más ancha. Ajustar correctamente la escala no equivale a ajustar la
distribución completa ni su dependencia condicional.

## Retroalimentación y próximos pasos

El avance más importante no es que un modelo haya ganado, sino haber construido
una comparación capaz de mostrar cuándo y por qué mi modelo pierde. El proyecto
ya incorpora proper scoring rules, inferencia pareada con bootstrap temporal,
Model Confidence Set, controles de look-ahead, linaje reproducible y 178 tests.

Los siguientes pasos son:

- igualar estrictamente el conjunto de información y la frecuencia de
  recalibración de todos los modelos;
- implementar portafolios MM completamente walk-forward antes de comparar
  resultados de inversión;
- evaluar benchmarks dinámicos adicionales, como cópulas, regímenes o
  volatilidad estocástica;
- reservar un holdout futuro realmente sellado para una evaluación
  confirmatoria.

El resultado actual es evidencia de desarrollo, no una recomendación de
inversión. El código y el protocolo están disponibles aquí:

https://github.com/tav0-m/mm-ipsa-research

#DataScience #QuantitativeFinance #TimeSeries #Python #OpenScience

---

## Orden de carga de las imágenes

1. `docs/assets/linkedin-v07/01-comparacion-modelos.png`
2. `docs/assets/linkedin-v07/02-estabilidad-temporal.png`
3. `docs/assets/linkedin-v07/03-inferencia-mm-vs-dcc.png`
4. `docs/assets/linkedin-v07/04-calibracion-distributiva.png`

## Texto alternativo sugerido

1. Tabla de cinco modelos y tres reglas de scoring; DCC-GARCH presenta la menor
   pérdida en CRPS, Energy y Variogram sobre 169 ventanas fuera de muestra.
2. Matriz de rangos CRPS por fold; DCC-GARCH gana 2023, 2025 y 2026 H1, mientras
   Student-t gana 2024.
3. Diferencias relativas MM-BCD menos DCC-GARCH con intervalos al 95%; los tres
   intervalos quedan sobre cero después de la corrección de Holm.
4. Dispersión frente a fiabilidad PIT; MM-BCD ajusta mejor la escala, mientras
   DCC-GARCH obtiene un menor desvío del histograma PIT.
