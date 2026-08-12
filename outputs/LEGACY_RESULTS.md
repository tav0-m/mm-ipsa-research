# Resultados legacy

Los CSV, NPY, figuras y snapshots anteriores a la migración de agosto de 2026 fueron generados con una función objetivo, gradientes, regularización, EWMA y benchmark diferentes de la versión actual.

No deben citarse como resultados de la investigación vigente. En particular, valores antiguos de F no son comparables porque cambió la normalización de momentos, se excluyó la diagonal de covarianza duplicada y la selección ahora usa G=F+lambda KL.

Para reemplazarlos:

```powershell
.\.venv\Scripts\python.exe run.py --step transform
.\.venv\Scripts\python.exe run.py --step mm
.\.venv\Scripts\python.exe run.py --step benchmarks
.\.venv\Scripts\python.exe run.py --step evaluate
.\.venv\Scripts\python.exe run.py --step portfolio
.\.venv\Scripts\python.exe run.py --step backtest
.\.venv\Scripts\python.exe run.py --step snapshot
.\.venv\Scripts\python.exe verify.py --scope full
```
