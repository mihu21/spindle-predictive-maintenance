[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$Input = "data\spindle_predictive_maintenance_10000_unlabeled.csv",
    [string]$OutputRoot = "output\prognostics_v6",
    [string]$ModelsRoot = "models",
    [int]$Limit = 0,
    [int]$ProgressEvery = 2500,
    [switch]$LegacyMLAudit,
    [switch]$DisableAnomalyAudit,
    [switch]$WithDatabase,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Require-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required file is missing: $Path"
    }
}

if ([string]::IsNullOrWhiteSpace($Python)) {
    if (Test-Path -LiteralPath ".\.venv-win\Scripts\python.exe") {
        $Python = (Resolve-Path ".\.venv-win\Scripts\python.exe").Path
    } else {
        $Python = "python"
    }
}

$env:OMP_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:NUMEXPR_NUM_THREADS = "2"
$env:PYTHONPATH = (Resolve-Path ".\src").Path

Write-Host "Using Python: $Python"
Write-Host "V6 primary method: probabilistic degradation-state / threshold-crossing prognostics."
Write-Host "No synthetic lifecycle generation or supervised model training is required."
Write-Host "Legacy ML is audit-only and disabled unless -LegacyMLAudit is supplied."

Require-File $Input
Require-File ".\config\prognostics.json"

if (-not $SkipTests) {
    & $Python -m compileall -q src tests main.py evaluate_prognostics_v6.py
    & $Python -m pytest -q
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$replay = Join-Path $OutputRoot "replay.csv"
$lifecycles = Join-Path $OutputRoot "lifecycles.csv"
$invalid = Join-Path $OutputRoot "invalid_rows.csv"
$database = Join-Path $OutputRoot "monitor.db"
$evaluation = Join-Path $OutputRoot "evaluation.json"

$replayArgs = @(
    "main.py", "replay",
    "--input", $Input,
    "--data-domain", "plant",
    "--models-root", $ModelsRoot,
    "--csv", $replay,
    "--lifecycles-csv", $lifecycles,
    "--invalid-csv", $invalid,
    "--database", $database,
    "--progress-every", "$ProgressEvery"
)
if (-not $DisableAnomalyAudit) { $replayArgs += "--anomaly-audit" }
if ($LegacyMLAudit) { $replayArgs += "--legacy-ml-audit" }
if (-not $WithDatabase) { $replayArgs += "--no-database" }
if ($Limit -gt 0) { $replayArgs += @("--limit", "$Limit") }

& $Python @replayArgs

& $Python evaluate_prognostics_v6.py `
    --replay $replay `
    --output $evaluation

Write-Host ""
Write-Host "V6 replay complete."
Write-Host "Replay:      $replay"
Write-Host "Evaluation:  $evaluation"
Write-Host "Lifecycles:  $lifecycles"
Write-Host "Invalid:     $invalid"
if ($WithDatabase) { Write-Host "SQLite:      $database" }
Write-Host ""
Write-Host "Important: evaluation on the existing one-machine trajectory is development/sanity evidence, not independent production validation."
