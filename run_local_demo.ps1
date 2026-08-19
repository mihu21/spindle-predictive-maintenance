param(
    [int]$Machines = 8,
    [double]$Hours = 72,
    [int]$Seed = 42,
    [string]$StartTime = "2026-01-01T00:00:00+00:00",
    [int]$CadenceMinutes = 10,
    [string]$Actor = "local-demo-user",
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Model = Join-Path $ProjectRoot "models\rul_v2_7_full_cadence.joblib"
$DemoDb = Join-Path $ProjectRoot "output\plant_shadow_demo\plant_shadow_demo.db"
$FrontendDir = Join-Path $ProjectRoot "frontend"
$FrontendModules = Join-Path $FrontendDir "node_modules"

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host " VVB001 Plant Shadow - Deterministic Local Demo" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Synthetic evidence only. Production authorization is disabled." -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Fail "Python virtual environment not found at: $Python"
}
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    Fail "Frozen v2.7 model not found at: $Model"
}
if (-not (Test-Path -LiteralPath $FrontendDir -PathType Container)) {
    Fail "Frontend directory not found at: $FrontendDir"
}
if (-not (Test-Path -LiteralPath $FrontendModules -PathType Container)) {
    Fail "Frontend dependencies are missing. Run setup_office_laptop.ps1 first."
}
$Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $Npm) {
    Fail "npm.cmd is not available in PATH."
}

Write-Host "[1/3] Rebuilding isolated demo evidence through the frozen runtime..." -ForegroundColor Green
& $Python main.py plant-shadow generate-demo `
    --database $DemoDb `
    --machines $Machines `
    --hours $Hours `
    --seed $Seed `
    --start-time $StartTime `
    --cadence-minutes $CadenceMinutes `
    --actor $Actor
if ($LASTEXITCODE -ne 0) {
    Fail "Demo generation failed. Backend and frontend were not started."
}
if (-not (Test-Path -LiteralPath $DemoDb -PathType Leaf)) {
    Fail "Demo generation completed without creating: $DemoDb"
}

Write-Host "[2/3] Starting the read-only FastAPI backend..." -ForegroundColor Green
$BackendCommand = "Set-Location '$ProjectRoot'; & '$Python' main.py plant-shadow serve-api --database '$DemoDb' --host 127.0.0.1 --port 8000"
$Backend = Start-Process powershell.exe -WindowStyle Hidden -PassThru -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $BackendCommand
)

$ApiReady = $false
for ($Attempt = 0; $Attempt -lt 20; $Attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/system/health" -TimeoutSec 2
        if ($Health.environment_mode -eq "DEMO") {
            $ApiReady = $true
            break
        }
    }
    catch {
        # Backend may still be importing the frozen runtime dependencies.
    }
}
if (-not $ApiReady) {
    Stop-Process -Id $Backend.Id -ErrorAction SilentlyContinue
    Fail "FastAPI did not become ready in DEMO mode on port 8000."
}

Write-Host "[3/3] Starting the existing React frontend..." -ForegroundColor Green
$NpmPath = $Npm.Source
$FrontendCommand = "Set-Location '$FrontendDir'; & '$NpmPath' run dev -- --host 127.0.0.1 --port 5173"
$Frontend = Start-Process powershell.exe -WindowStyle Hidden -PassThru -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $FrontendCommand
)

Start-Sleep -Seconds 2
if (-not $SkipBrowser) {
    Start-Process "http://127.0.0.1:5173"
}

Write-Host ""
Write-Host "LOCAL DEMO MODE is running." -ForegroundColor Cyan
Write-Host "API:      http://127.0.0.1:8000" -ForegroundColor Gray
Write-Host "Frontend: http://127.0.0.1:5173" -ForegroundColor Gray
Write-Host "Backend PID:  $($Backend.Id)" -ForegroundColor DarkGray
Write-Host "Frontend PID: $($Frontend.Id)" -ForegroundColor DarkGray
Write-Host "Database: $DemoDb" -ForegroundColor DarkGray
Write-Host ""
Write-Host "No PostgreSQL ingestion was started and no DSN is required." -ForegroundColor Yellow

