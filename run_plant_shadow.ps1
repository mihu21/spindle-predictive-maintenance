param(
    [string]$Actor = "local-user",
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"

# Project root = directory containing this script.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$FrontendDir = Join-Path $ProjectRoot "frontend"
$FrontendModules = Join-Path $FrontendDir "node_modules"
$LocalConfig = Join-Path $ProjectRoot "config\plant_shadow_sources.local.json"
$ExampleConfig = Join-Path $ProjectRoot "config\plant_shadow_sources.example.json"
$EvidenceDb = Join-Path $ProjectRoot "output\plant_shadow\plant_shadow.db"

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " VVB001 Plant Shadow - One Command Launcher" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# --- Prerequisite checks ---
if (-not (Test-Path $Python)) {
    Fail "Python virtual environment not found at: $Python"
}

if (-not (Test-Path $FrontendDir)) {
    Fail "Frontend directory not found at: $FrontendDir"
}

$Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $Npm) {
    Fail "npm is not available in PATH. Run setup_office_laptop.ps1 after installing Node.js LTS."
}
$NpmPath = $Npm.Source

if (-not (Test-Path $FrontendModules)) {
    Fail "Frontend dependencies are missing. Run setup_office_laptop.ps1 first."
}

# --- Ensure local source config exists ---
if (-not (Test-Path $LocalConfig)) {
    if (Test-Path $ExampleConfig) {
        Copy-Item $ExampleConfig $LocalConfig
        Write-Host "Created:" -ForegroundColor Yellow
        Write-Host "  config\plant_shadow_sources.local.json"
        Write-Host ""
        Write-Host "Edit that file to match the office database, set the required VVB001_*_DSN" -ForegroundColor Yellow
        Write-Host "environment variable(s), then run this script again." -ForegroundColor Yellow
        exit 0
    }
    else {
        Fail "Neither local nor example plant source configuration exists."
    }
}

# --- Initialize the local plant-shadow evidence DB if necessary ---
if (-not (Test-Path $EvidenceDb)) {
    Write-Host "[1/4] Initializing local plant-shadow evidence database..." -ForegroundColor Green

    & $Python main.py plant-shadow source-save `
        --sources $LocalConfig `
        --actor $Actor

    if ($LASTEXITCODE -ne 0) {
        Fail "plant-shadow source-save failed. Check the source config and DSN environment variables."
    }

    if (-not (Test-Path $EvidenceDb)) {
        Fail "source-save completed but plant_shadow.db was not created."
    }

    Write-Host "      Evidence database initialized." -ForegroundColor Green
}
else {
    Write-Host "[1/4] Evidence database already exists." -ForegroundColor Green
}

# --- Quick status check ---
Write-Host "[2/4] Checking plant-shadow status..." -ForegroundColor Green
& $Python main.py plant-shadow status
if ($LASTEXITCODE -ne 0) {
    Fail "plant-shadow status failed."
}

# --- Start backend in its own PowerShell window ---
Write-Host "[3/4] Starting backend on http://127.0.0.1:8000 ..." -ForegroundColor Green

$BackendCommand = @"
Set-Location '$ProjectRoot'
Write-Host 'VVB001 Plant Shadow Backend' -ForegroundColor Cyan
& '$Python' main.py plant-shadow serve-api --host 127.0.0.1
"@

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $BackendCommand
)

# Give Uvicorn a moment to start.
Start-Sleep -Seconds 2

# --- Start frontend in its own PowerShell window ---
Write-Host "[4/4] Starting frontend on http://127.0.0.1:5173 ..." -ForegroundColor Green

$FrontendCommand = @"
Set-Location '$FrontendDir'
Write-Host 'VVB001 Plant Shadow Frontend' -ForegroundColor Cyan
& '$NpmPath' run dev
"@

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $FrontendCommand
)

Start-Sleep -Seconds 2

if (-not $SkipBrowser) {
    Start-Process "http://127.0.0.1:5173"
}

Write-Host ""
Write-Host "Plant Shadow launched." -ForegroundColor Cyan
Write-Host "Backend : http://127.0.0.1:8000" -ForegroundColor Gray
Write-Host "Frontend: http://127.0.0.1:5173" -ForegroundColor Gray
Write-Host ""
Write-Host "For the first office database read, run this separately:" -ForegroundColor Yellow
Write-Host "  .venv\Scripts\python.exe main.py plant-shadow ingest --sources config\plant_shadow_sources.local.json --once"
Write-Host ""
Write-Host "This launcher intentionally does NOT start continuous ingestion." -ForegroundColor Yellow
