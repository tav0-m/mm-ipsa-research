param(
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "No existe .venv. Cree el entorno con: py -3.11 -m venv .venv"
}

Push-Location $ProjectRoot
try {
    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "pip check fallo" }

    & $Python -m compileall -q src run.py verify.py
    if ($LASTEXITCODE -ne 0) { throw "compileall fallo" }

    & $Python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "La suite unittest fallo" }

    if ($Full) {
        & $Python verify.py --scope full
        if ($LASTEXITCODE -ne 0) { throw "La verificacion integral fallo" }
    }
}
finally {
    Pop-Location
}

Write-Host "Release check OK" -ForegroundColor Green
