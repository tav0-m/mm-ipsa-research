# Changelog

Todos los cambios relevantes de este proyecto se documentarán en este archivo.

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
