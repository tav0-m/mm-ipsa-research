param(
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SourceRoot = Join-Path $ProjectRoot "src"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "No existe .venv. Cree el entorno con: py -3.11 -m venv .venv"
}

Push-Location $ProjectRoot
$PreviousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $SourceRoot

    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "pip check fallo" }

    & $Python -m compileall -q src run.py verify.py
    if ($LASTEXITCODE -ne 0) { throw "compileall fallo" }

    & $Python -m ruff check src tests run.py verify.py
    if ($LASTEXITCODE -ne 0) { throw "ruff fallo" }

    & $Python -m pyright --project pyrightconfig.json
    if ($LASTEXITCODE -ne 0) { throw "pyright fallo" }

    & $Python -m mm_ipsa --version
    if ($LASTEXITCODE -ne 0) { throw "El entrypoint mm_ipsa fallo" }

    & $Python -m mm_ipsa run --step all --plan
    if ($LASTEXITCODE -ne 0) { throw "El plan de ejecucion fallo" }

    & $Python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "La suite unittest fallo" }

    if ($Full) {
        & $Python -m mm_ipsa verify --scope full
        if ($LASTEXITCODE -ne 0) { throw "La verificacion integral fallo" }
    }
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}

Write-Host "Release check OK" -ForegroundColor Green
