# MM-IPSA Research

[![CI](https://github.com/tav0-m/mm-ipsa-research/actions/workflows/ci.yml/badge.svg)](https://github.com/tav0-m/mm-ipsa-research/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/Python-3.11%E2%80%933.12-3776AB.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Resumen de MM-IPSA Research](docs/assets/linkedin-project-card.png)

Plataforma de investigación cuantitativa independiente para estudiar generación de escenarios discretos por ajuste de momentos y su utilidad en decisiones de portafolio sobre acciones chilenas.

La pregunta no es si MM-BCD reproduce media, covarianza y momentos superiores —lo hace con alta precisión—, sino si esa calibración mejora pronósticos probabilísticos y decisiones económicas fuera de muestra frente a controles Gaussian, Student-t e histórico EWMA.

**Versión pública actual:** `v0.6.0` · **Estado:** validación de desarrollo · **No es asesoría de inversión.**

## Resultado principal

La complejidad no produjo una superioridad general. En validación rolling-origin con cuatro folds, 169 ventanas no solapadas de cinco días y recalibración completa de todos los modelos en cada origen:

| Modelo | CRPS pooled | Energy Score | Variogram Score | Folds ganados en CRPS |
|---|---:|---:|---:|---:|
| Student-t | **0.020904** | **0.100780** | **0.654249** | 3/4 |
| Gaussiano | 0.020934 | 0.100845 | 0.661880 | 1/4 |
| Histórico EWMA | 0.020969 | 0.101302 | 0.654289 | 0/4 |
| MM-BCD | 0.020990 | 0.101292 | 0.654383 | 0/4 |

MM frente al Gaussiano en CRPS obtuvo una diferencia de `+0.000056`, IC95 `[-0.000019, 0.000138]` y `p_Holm=0.595`: no hay diferencia estadísticamente distinguible. El Model Confidence Set al 95% deja a MM **fuera** en CRPS y Energy Score, y **dentro** en Variogram Score, donde además supera al Gaussiano de forma distinguible (`-0.007497`, `p_Holm=0.004`).

El diagnóstico de calibración identifica el mecanismo: MM alcanza la mejor razón de dispersión de los cuatro modelos (`0.995` contra un ideal de `1.000`) y a la vez el histograma PIT menos uniforme. Ajustar los cuatro primeros momentos no equivale a ajustar la distribución, y las reglas de scoring propias evalúan la forma completa.

> **Corrección metodológica respecto de v0.5.0.** Los grados de libertad del Student-t eran una constante no estimada (`6.0`). Al estimarlos por verosimilitud en cada origen —el rango resultante es 12 a 25— tres conclusiones de v0.5.0 dejan de sostenerse: la ventaja de MM en Energy Score frente al histórico y sus desventajas en Variogram frente a Student-t e histórico. El detalle está en [research/RESULTS_20260813.md](research/RESULTS_20260813.md).

![Estabilidad temporal de CRPS](docs/assets/rolling-origin-crps.png)

![Diferencias pareadas de scores](docs/assets/paired-score-differences.png)

## Diseño de investigación

```mermaid
flowchart LR
    A["Precios y máscara raw"] --> B["Calidad antes de imputar"]
    B --> C["Retornos y ventanas H=5"]
    C --> D["Targets EWMA por fold"]
    D --> E["MM-BCD"]
    D --> F["Gaussian / Student-t / histórico"]
    E --> G["CRPS / Energy / Variogram / VaR"]
    F --> G
    G --> H["Bootstrap temporal + Holm"]
    E --> I["Portafolios + costos"]
    I --> J["Linaje y snapshot SHA-256"]
    H --> J
```

- Universo principal: 15 acciones chilenas.
- Datos locales actuales: 2020-01-02 a 2026-06-10.
- Horizonte: retorno terminal compuesto a cinco días.
- Rolling-origin expansivo: 2023, 2024, 2025 y 2026-H1.
- Todos los modelos se recalibran en cada fold usando solo datos anteriores, incluidos los grados de libertad del Student-t y la contracción de covarianza.
- Inferencia: moving-block bootstrap de 5.000 muestras con ancho de bloque elegido por Politis-White, corrección Holm sobre nueve contrastes y Diebold-Mariano con varianza HAC como verificación independiente.
- Model Confidence Set al 95% para identificar qué modelos no son descartables como óptimos.
- Diagnósticos de calibración PIT con soporte igualado entre modelos.
- Sensibilidad separada de liquidez seleccionada exclusivamente con métricas in-sample.
- 159 pruebas automatizadas, Ruff y Pyright sin errores, y nueve etapas de linaje verificadas.

El protocolo completo está en [research/PROTOCOL.md](research/PROTOCOL.md) y los cortes rolling-origin están congelados en [research/rolling_origin.yaml](research/rolling_origin.yaml).

## Arquitectura de software

```text
src/mm_ipsa/
├── analysis/       # rolling-origin, liquidez y activos públicos
├── backtest/       # simulación walk-forward y costos
├── data/           # descarga, calidad y transformación temporal
├── evaluation/     # scoring rules, inferencia pareada, MCS y calibración
├── mm/             # objetivo, gradientes, BCD y diagnósticos
├── models/         # controles Gaussian, Student-t e histórico EWMA
├── portfolio/      # optimización y baselines robustos
├── cli.py          # comando público mm-ipsa
├── pipeline.py     # orquestación, reanudación y snapshots
└── verification.py # contratos científicos y de linaje
```

`research/` conserva el protocolo y el informe; `docs/` contiene únicamente la ficha pública; `tests/` verifica contratos matemáticos, temporales y operacionales. Los datos y resultados derivados permanecen fuera de Git.

## Inicio rápido en PowerShell

Requisitos: Python 3.11 o 3.12 y Windows PowerShell.

```powershell
git clone https://github.com/tav0-m/mm-ipsa-research.git
cd mm-ipsa-research
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\scripts\release_check.ps1
```

Pipeline completo con una nueva descarga:

```powershell
.\.venv\Scripts\mm-ipsa.exe run --step all
```

Si ya existe una descarga local íntegra y no se desea consultar nuevamente al proveedor:

```powershell
.\.venv\Scripts\mm-ipsa.exe run --step reuse-download
.\.venv\Scripts\mm-ipsa.exe run --step all --resume
```

`--resume` omite únicamente etapas cuyo manifiesto, entradas y salidas conservan sus hashes. Para inspeccionar qué se ejecutaría sin modificar artefactos:

```powershell
.\.venv\Scripts\mm-ipsa.exe run --step all --plan
```

La validación rolling-origin tarda aproximadamente dos minutos en la máquina de desarrollo; es una verificación de release, no un chequeo interactivo rápido.

## Verificación

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\mm-ipsa.exe verify --scope full
```

Un test verde prueba contratos de software y trazabilidad; no prueba rentabilidad futura. Los datos raw y los artefactos derivados no se distribuyen en Git. Consulta [DATA_POLICY.md](DATA_POLICY.md) y [ROADMAP.md](ROADMAP.md) para conocer los límites y siguientes etapas.

## Documentación

- [Informe de investigación en PDF](research/build/MM_Research_Report.pdf)
- [Fuente LaTeX del informe](research/MM_Research_Report.tex)
- [Resultados actuales](research/RESULTS_20260813.md)
- [Resultados de v0.5.0, superados](research/RESULTS_20260810.md)
- [Guía de implementación](research/IMPLEMENTATION_GUIDE.md)
- [Referencias](research/REFERENCES.md)
- [Ficha pública](docs/index.html)
- [Borrador para LinkedIn](docs/LINKEDIN_POST_ES.md)
- [Historial de versiones](CHANGELOG.md)

## Licencia

Código y documentación propia bajo licencia MIT. Los datos de mercado conservan los términos y restricciones de su proveedor original.
