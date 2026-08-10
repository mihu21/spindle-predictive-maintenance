$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$root = (Resolve-Path $PSScriptRoot).Path
$packageRoot = Join-Path $root "packages"
$sourceStage = Join-Path $packageRoot ".source_stage"
$evidenceStage = Join-Path $packageRoot ".evidence_stage"
$sourceZip = Join-Path $packageRoot "spindle_maintenance_source_reproduction.zip"
$evidenceZip = Join-Path $packageRoot "spindle_maintenance_evidence_bundle.zip"

New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null
if (Test-Path -LiteralPath $sourceStage) { Remove-Item -LiteralPath $sourceStage -Recurse -Force }
if (Test-Path -LiteralPath $evidenceStage) { Remove-Item -LiteralPath $evidenceStage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $sourceStage | Out-Null
New-Item -ItemType Directory -Force -Path $evidenceStage | Out-Null

function Copy-Relative([string]$relative) {
    $source = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $source)) { return }
    $destination = Join-Path $sourceStage $relative
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    if ((Get-Item -LiteralPath $source).PSIsContainer) {
        Copy-Item -LiteralPath $source -Destination $parent -Recurse -Force
    } else {
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
}

@(
    ".gitignore", "requirements.txt", "requirements-lock.txt", "main.py", "evaluate.py", "validate_compare.py",
    "README.md", "IMPLEMENTATION_REPORT.md", "spindle_model_validation_fix_spec.md",
    "reproduce.ps1", "build_packages.ps1", "build_portable_zip.py", "augment_environment_metadata.py", "verify_final_artifacts.py", "verify_anomaly_corrections.py", "config", "src", "tests", "docs",
    "data\spindle_predictive_maintenance_10000_unlabeled.csv",
    "data\spindle_predictive_maintenance_mock.csv",
    "data\realistic_spindle_mock_metadata.json",
    "data\profile_validation_cases",
    "data\anomaly_fixtures",
    "models\mock\candidate", "models\mock\registry_audit.jsonl",
    "models\realistic\candidate", "models\realistic\registry_audit.jsonl",
    "models\plant\.gitkeep",
    "output\test_output.txt", "output\compile_output.txt",
    "output\anomaly_test_output.txt", "output\anomaly_evaluation.json",
    "output\anomaly_correction_test_output.txt", "output\anomaly_correction_verification.json",
    "output\anomaly_training_screening.json", "output\plant_anomaly_summary.json",
    "output\plant_anomaly_intervals.json", "output\plant_anomaly_intervals.csv",
    "output\model_hash_comparison_anomaly_correction.json",
    "output\model_environment_update.json", "output\final_verification.json", "output\profile_cases",
    "output\realistic\actual_data_profile.json", "output\realistic\model_metrics.json",
    "output\realistic\model_evaluation.json", "output\realistic\guardrail_evaluation.json",
    "output\realistic\promotion_refusal.json", "output\realistic\lifecycles.csv",
    "output\realistic\invalid_rows.csv",
    "output\mock\model_metrics.json", "output\mock\model_evaluation.json",
    "output\mock\promotion_refusal.json", "output\mock\lifecycles.csv",
    "output\mock\invalid_rows.csv"
) | ForEach-Object { Copy-Relative $_ }

Get-ChildItem -LiteralPath $sourceStage -Recurse -Directory -Force |
    Where-Object { $_.Name -eq "__pycache__" } |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $sourceStage -Recurse -File -Force |
    Where-Object { $_.Extension -eq ".pyc" -or $_.Name -match "\.tmp$|~$" } |
    Remove-Item -Force

if (Test-Path -LiteralPath $sourceZip) { Remove-Item -LiteralPath $sourceZip -Force }
$python = Join-Path $root ".venv-win\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
& $python (Join-Path $root "build_portable_zip.py") --root $sourceStage --output $sourceZip
if ($LASTEXITCODE -ne 0) { throw "Portable source ZIP build failed" }

$evidenceFiles = @(
    "data\realistic_spindle_mock.csv",
    "output\realistic\replay.csv", "output\realistic\monitor.db", "output\realistic\training.db",
    "output\mock\replay.csv", "output\mock\monitor.db", "output\mock\training.db",
    "output\plant_anomaly_replay.csv", "output\plant_anomaly.db",
    "output\plant_anomaly_events.json", "output\plant_anomaly_intervals.json",
    "output\plant_anomaly_intervals.csv", "output\plant_anomaly_summary.json"
)
foreach ($relative in $evidenceFiles) {
    $source = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $source)) { continue }
    $destination = Join-Path $evidenceStage $relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}
if (Test-Path -LiteralPath $evidenceZip) { Remove-Item -LiteralPath $evidenceZip -Force }
& $python (Join-Path $root "build_portable_zip.py") --root $evidenceStage --output $evidenceZip
if ($LASTEXITCODE -ne 0) { throw "Portable evidence ZIP build failed" }

$artifacts = @($sourceZip, $evidenceZip) | ForEach-Object {
    $item = Get-Item -LiteralPath $_
    $archive = [System.IO.Compression.ZipFile]::OpenRead($item.FullName)
    try {
        $entryCount = $archive.Entries.Count
        $nonPortable = @($archive.Entries | Where-Object { $_.FullName.Contains("\") })
        if ($nonPortable.Count -ne 0) {
            throw "Archive contains non-portable backslash entry names: $($nonPortable[0].FullName)"
        }
    } finally { $archive.Dispose() }
    [ordered]@{
        file_name = $item.Name
        size_bytes = $item.Length
        entry_count = $entryCount
        portable_entry_names = $true
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifest = [ordered]@{
    generated_at = (Get-Date).ToString("o")
    artifacts = $artifacts
}
$manifestJson = $manifest | ConvertTo-Json -Depth 5
$manifestJson | Set-Content -LiteralPath (Join-Path $root "output\package_manifest.json") -Encoding UTF8
$manifestJson | Set-Content -LiteralPath (Join-Path $packageRoot "package_manifest.json") -Encoding UTF8
Remove-Item -LiteralPath $sourceStage -Recurse -Force
Remove-Item -LiteralPath $evidenceStage -Recurse -Force
$manifestJson
