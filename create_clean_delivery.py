"""Build a source-only delivery archive with portable POSIX entry names."""
from __future__ import annotations
import shutil, tempfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = Path(tempfile.gettempdir()) / "spindle_forecast_policy_stage_20260806_009"
ZIP = ROOT / "spindle_forecast_policy_clean_20260806_009.zip"
ITEMS = ["src", "tests", "config", "docs", "README.md", "IMPLEMENTATION_REPORT.md", "DELIVERY_MANIFEST.txt",
 "evaluate_monitoring.py", "create_clean_delivery.py", "main.py", "evaluate.py", "evaluate_fp_rate.py", "validate_compare.py",
 "augment_environment_metadata.py", "verify_anomaly_corrections.py", "verify_final_artifacts.py", "requirements.txt", "requirements-lock.txt", "reproduce.ps1", "build_portable_zip.py", "build_packages.ps1", "spindle_model_validation_fix_spec.md", ".gitignore"]
BAD_PARTS = {".venv", ".venv-win", "venv", ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache", "output", "outputs"}
BAD_SUFFIXES = {".zip", ".log", ".db", ".sqlite", ".sqlite3", ".tmp", ".bak", ".pyc"}

if STAGE.exists(): shutil.rmtree(STAGE)
STAGE.mkdir()
for item in ITEMS:
    src, dst = ROOT / item, STAGE / item
    if src.is_dir(): shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else: shutil.copy2(src, dst)
files = [p for p in STAGE.rglob("*") if p.is_file()]
bad = [p for p in files if any(part in BAD_PARTS for part in p.relative_to(STAGE).parts) or p.suffix.lower() in BAD_SUFFIXES or p.stat().st_size > 10 * 1024 * 1024]
if bad: raise SystemExit("disallowed staged files: " + ", ".join(str(p) for p in bad))
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(STAGE).as_posix())
with zipfile.ZipFile(ZIP) as archive:
    names = archive.namelist()
    if not names or any("\\" in name for name in names): raise SystemExit("invalid archive entries")
print(f"{ZIP}\nfiles={len(names)}\nsize={ZIP.stat().st_size}")
