# Política de datos

Este repositorio no redistribuye precios raw ni artefactos derivados de mercado.

- Los datos se descargan bajo los términos del proveedor configurado por cada usuario.
- `outputs/` está excluido de Git y se regenera desde configuración, código y datos autorizados.
- Cada descarga guarda cobertura previa a imputación, máscara de observación, parámetros de solicitud y hashes SHA-256.
- El pipeline no utiliza un fallback sintético cuando la descarga o la calidad fallan.
- Los activos de `docs/assets/` contienen únicamente métricas agregadas y visualizaciones, no series de precios ni retornos por activo.

Yahoo Finance resulta adecuado para investigación exploratoria reproducible, pero no sustituye una fuente institucional para decisiones de inversión o un producto comercial.
