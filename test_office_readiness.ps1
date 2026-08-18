param(
    [switch]$Full,
    [switch]$SkipFrontend,
    [switch]$IncludeDatabase,
    [string]$Actor = $env:USERNAME
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$FrontendDir = Join-Path $ProjectRoot "frontend"
$LocalConfig = Join-Path $ProjectRoot "config\plant_shadow_sources.local.json"
$Model = Join-Path $ProjectRoot "models\rul_v2_7_full_cadence.joblib"
$Manifest = Join-Path $ProjectRoot "output\plant_shadow\plant_shadow_runtime_manifest.json"
$TestTemp = Join-Path $ProjectRoot ".pytest_tmp_office_readiness"
$ExpectedModelHash = "ecad8f4f704129a3f0456c3c8dd47aabeb94ea6dfbdfd4c264313aea46076b9a"

Set-Location $ProjectRoot

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Invoke-Checked([string]$Label, [scriptblock]$Action) {
    Write-Host $Label -ForegroundColor Green
    & $Action
    if ($LASTEXITCODE -ne 0) {
        Fail "$Label failed with exit code $LASTEXITCODE."
    }
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " VVB001 Office Readiness Verification" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Mode: observational plant shadow; production authorization remains false." -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path $Python)) {
    Fail "Virtual environment not found. Run .\setup_office_laptop.ps1 first."
}
if (-not (Test-Path $Model)) {
    Fail "Frozen v2.7 model artifact is missing: $Model"
}
if (-not (Test-Path $Manifest)) {
    Fail "Runtime manifest is missing: $Manifest"
}

& $Python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)"
if ($LASTEXITCODE -ne 0) {
    Fail "The office test environment must use Python 3.14. Rerun setup_office_laptop.ps1 with a Python 3.14 installation."
}
$env:LOKY_MAX_CPU_COUNT = [string][Math]::Max(1, [Environment]::ProcessorCount)

Invoke-Checked "[1/6] Checking Python and installed dependencies..." {
    & $Python --version
    if ($LASTEXITCODE -eq 0) {
        & $Python -m pip check
    }
}

$ActualModelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Model).Hash.ToLowerInvariant()
if ($ActualModelHash -ne $ExpectedModelHash) {
    Fail "Frozen model SHA-256 mismatch. Expected $ExpectedModelHash but found $ActualModelHash."
}
Write-Host "[2/6] Frozen model hash matches the accepted v2.7 artifact." -ForegroundColor Green

Invoke-Checked "[3/6] Verifying runtime manifest and golden replay..." {
    & $Python -c "import sys; sys.path.insert(0, 'src'); from vvb001_monitor.plant_shadow.manifest import verify_manifest; print(verify_manifest('.', 'output/plant_shadow/plant_shadow_runtime_manifest.json')['deployment_id'])"
    if ($LASTEXITCODE -eq 0) {
        & $Python main.py plant-shadow verify-golden
    }
}

$PytestArgs = @("-m", "pytest", "-q", "--disable-warnings", "--basetemp", $TestTemp)
if (-not $Full) {
    $PytestArgs += @(
        "tests\test_config.py",
        "tests\test_readonly_contract.py",
        "tests\test_plant_shadow_contracts.py",
        "tests\test_plant_shadow_source.py",
        "tests\test_plant_shadow_storage.py",
        "tests\test_plant_shadow_service.py",
        "tests\test_plant_shadow_operating_context.py",
        "tests\test_vibration_operating.py",
        "tests\test_plant_shadow_api.py",
        "tests\test_plant_shadow_evaluation.py",
        "tests\test_plant_shadow_manifest_cli.py",
        "tests\test_plant_shadow_golden.py",
        "tests\test_plant_shadow_native_postgres.py",
        "tests\test_native_postgres_lifecycle.py"
    )
}

$SuiteName = if ($Full) { "full Python suite" } else { "office plant-shadow suite" }
Invoke-Checked "[4/6] Running $SuiteName with project-local test scratch space..." {
    & $Python @PytestArgs
}

if (-not $SkipFrontend) {
    $Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($null -eq $Npm) {
        Fail "npm is unavailable. Install Node.js LTS or rerun setup_office_laptop.ps1."
    }
    if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
        Fail "Frontend dependencies are absent. Run .\setup_office_laptop.ps1 first."
    }
    Push-Location $FrontendDir
    try {
        Invoke-Checked "[5/6] Testing and rebuilding the dashboard..." {
            & $Npm.Source test
            if ($LASTEXITCODE -eq 0) {
                & $Npm.Source run build
            }
        }
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Host "[5/6] Frontend checks skipped by request." -ForegroundColor Yellow
}

if ($IncludeDatabase) {
    if (-not (Test-Path $LocalConfig)) {
        Fail "Local source config is missing. Copy config\plant_shadow_sources.example.json to config\plant_shadow_sources.local.json and edit it."
    }
    try {
        $SourceDocument = Get-Content -LiteralPath $LocalConfig -Raw | ConvertFrom-Json
    }
    catch {
        Fail "Local source config is not valid JSON: $($_.Exception.Message)"
    }
    foreach ($Source in @($SourceDocument.sources)) {
        $DsnName = [string]$Source.dsn_env
        if ([string]::IsNullOrWhiteSpace($DsnName)) {
            Fail "Every source must name a dsn_env environment variable."
        }
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($DsnName, "Process"))) {
            Fail "Required secret environment variable '$DsnName' is not set in this PowerShell session."
        }
    }
    if ([string]::IsNullOrWhiteSpace($Actor)) {
        Fail "Actor is required for the local source audit record."
    }
    Invoke-Checked "[6/6] Saving the redacted source, performing one bounded read-only ingest, and showing status..." {
        & $Python main.py plant-shadow source-save --sources $LocalConfig --actor $Actor
        if ($LASTEXITCODE -eq 0) {
            & $Python main.py plant-shadow ingest --sources $LocalConfig --once
        }
        if ($LASTEXITCODE -eq 0) {
            & $Python main.py plant-shadow status
        }
    }
}
else {
    Write-Host "[6/6] Live database check not requested; no database connection was attempted." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "OFFICE READINESS: PASS" -ForegroundColor Cyan
if (-not $IncludeDatabase) {
    Write-Host "Offline software checks passed. Database/schema/index/credential readiness remains to be proven with -IncludeDatabase." -ForegroundColor Yellow
}
