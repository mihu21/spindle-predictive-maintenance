from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.getenv("VVB001_RUN_NATIVE_POSTGRES_E2E") != "1",
    reason="set VVB001_RUN_NATIVE_POSTGRES_E2E=1 to run the native local PostgreSQL E2E",
)
def test_native_multisource_postgres_e2e():
    root = Path(__file__).parents[1]
    subprocess.run(
        [sys.executable, str(root / "tests/e2e/run_plant_shadow_postgres_e2e.py")],
        cwd=root,
        check=True,
        timeout=600,
    )
