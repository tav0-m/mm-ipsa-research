# Publicación para LinkedIn

Convertí una investigación cuantitativa en una plataforma reproducible para responder una pregunta sencilla, pero incómoda:

**¿Ajustar casi exactamente los primeros cuatro momentos y la covarianza produce mejores escenarios financieros fuera de muestra?**

Implementé MM-BCD y lo comparé con modelos Gaussiano, Student-t e histórico EWMA sobre 15 acciones chilenas.

La versión 0.5 incluye:

- cuatro folds rolling-origin y 169 ventanas OOS de cinco días;
- CRPS, Energy Score y Variogram Score;
- bootstrap temporal y corrección de Holm;
- un paquete Python con CLI y ejecución reanudable por hashes;
- 49 tests, Ruff y Pyright sin errores, CI multiversión, nueve etapas de linaje y snapshots SHA-256.

El resultado fue más valioso que una victoria artificial: **MM-BCD reprodujo los momentos con alta precisión, pero no mostró superioridad predictiva general**. Student-t obtuvo el mejor CRPS pooled y MM ganó 0 de 4 folds en esa métrica.

Mi principal aprendizaje: en investigación cuantitativa, optimizar mejor el objetivo in-sample no implica necesariamente pronosticar mejor ni tomar mejores decisiones fuera de muestra.

¿Qué benchmark o prueba de robustez agregarías para someter este resultado a una evaluación todavía más exigente?

Código, protocolo, informe y resultados:

https://github.com/tav0-m/mm-ipsa-research

https://tav0-m.github.io/mm-ipsa-research/

#DataScience #QuantitativeFinance #TimeSeries #Python #OpenScience

Este proyecto es investigación independiente y no constituye asesoría de inversión.

## Material recomendado

1. Para una publicación sencilla, use únicamente `docs/assets/linkedin-01-cover.png`.
2. Para una publicación con evidencia, cargue estas cinco imágenes en este orden:
   - `docs/assets/linkedin-01-cover.png`
   - `docs/assets/linkedin-02-pooled-crps.png`
   - `docs/assets/linkedin-03-temporal-crps.png`
   - `docs/assets/linkedin-04-paired-inference.png`
   - `docs/assets/linkedin-05-calibration-vs-prediction.png`
3. Confirme que GitHub Actions y GitHub Pages están verdes antes de publicar.
