param(
    [switch]$SkipFrontend,
    [switch]$SkipVerification
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$FrontendDir = Join-Path $ProjectRoot "frontend"
$LocalConfig = Join-Path $ProjectRoot "config\plant_shadow_sources.local.json"
$ExampleConfig = Join-Path $ProjectRoot "config\plant_shadow_sources.example.json"
$ReadinessScript = Join-Path $ProjectRoot "test_office_readiness.ps1"

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
Write-Host " VVB001 Office Laptop Setup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $Python)) {
    $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        & $PyLauncher.Source -3.14 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)"
        if ($LASTEXITCODE -eq 0) {
            Invoke-Checked "[1/6] Creating Python 3.14 virtual environment..." {
                & $PyLauncher.Source -3.14 -m venv $VenvDir
            }
        }
    }

    if (-not (Test-Path $Python)) {
        $SystemPython = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($null -ne $SystemPython) {
            & $SystemPython.Source -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)"
            if ($LASTEXITCODE -eq 0) {
                Invoke-Checked "[1/6] Creating Python 3.14 virtual environment..." {
                    & $SystemPython.Source -m venv $VenvDir
                }
            }
        }
    }
}
else {
    Write-Host "[1/6] Existing virtual environment found." -ForegroundColor Green
}

if (-not (Test-Path $Python)) {
    Fail "Python 3.14 is required. Install the 64-bit Python 3.14 release, then run this script again."
}

& $Python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)"
if ($LASTEXITCODE -ne 0) {
    Fail "The existing .venv is not Python 3.14. Rename or remove it, then rerun this script to create the required environment."
}

Invoke-Checked "[2/6] Installing Python test/runtime dependencies..." {
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -eq 0) {
        & $Python -m pip install -r requirements-dev.txt -c requirements-office-lock.txt
    }
}

Invoke-Checked "[3/6] Checking the Python environment..." {
    & $Python -m pip check
}

if (-not $SkipFrontend) {
    if (-not (Test-Path $FrontendDir)) {
        Fail "Frontend directory not found at: $FrontendDir"
    }
    $Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($null -eq $Npm) {
        Fail "Node.js LTS/npm is required for the dashboard. Install it, or rerun with -SkipFrontend for backend-only preparation."
    }
    $Node = Get-Command node.exe -ErrorAction SilentlyContinue
    if ($null -eq $Node) {
        Fail "node.exe is not available in PATH. Repair the Node.js LTS installation."
    }
    & $Node.Source -e "const [a,b]=process.versions.node.split('.').map(Number); process.exit((a===20&&b>=19)||(a===22&&b>=12)||a>22 ? 0 : 1)"
    if ($LASTEXITCODE -ne 0) {
        Fail "The dashboard requires Node.js 20.19+, 22.12+, or a newer current release. Install a current Node.js LTS release."
    }
    Push-Location $FrontendDir
    try {
        Invoke-Checked "[4/6] Installing locked frontend dependencies..." {
            & $Npm.Source ci
        }
        Invoke-Checked "[5/6] Testing and building the frontend..." {
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
    Write-Host "[4/6] Frontend setup skipped by request." -ForegroundColor Yellow
    Write-Host "[5/6] Frontend verification skipped by request." -ForegroundColor Yellow
}

if (-not (Test-Path $LocalConfig)) {
    if (-not (Test-Path $ExampleConfig)) {
        Fail "Plant source example configuration is missing."
    }
    Copy-Item -LiteralPath $ExampleConfig -Destination $LocalConfig
    Write-Host "Created ignored local source template: config\plant_shadow_sources.local.json" -ForegroundColor Yellow
}

if (-not $SkipVerification) {
    $ReadinessArgs = @("-SkipFrontend")
    Invoke-Checked "[6/6] Running offline office-readiness verification..." {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ReadinessScript @ReadinessArgs
    }
}
else {
    Write-Host "[6/6] Offline readiness verification skipped by request." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Office laptop setup is complete." -ForegroundColor Cyan
Write-Host "Next: edit config\plant_shadow_sources.local.json and set the DSN environment variable named by dsn_env." -ForegroundColor Yellow
Write-Host "Then run: .\test_office_readiness.ps1 -IncludeDatabase -Actor `"your-name`"" -ForegroundColor Yellow
