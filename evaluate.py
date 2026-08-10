from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.config import load_config
from spindle_monitor.io import read_csv_readings
from spindle_monitor.monitor import ConditionMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate configured rules on a labelled CSV")
    parser.add_argument(
        "--input",
        default="data/spindle_predictive_maintenance_10000_unlabeled.csv",
    )
    parser.add_argument("--config", default="config/thresholds.json")
    args = parser.parse_args()

    config = load_config(args.config)
    monitor = ConditionMonitor(config)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    effective_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    labelled = 0
    raw_correct = 0
    effective_correct = 0
    raw_changes = 0
    effective_changes = 0
    previous_raw = None
    previous_effective = None

    for reading, label in read_csv_readings(args.input):
        result = monitor.process(reading)
        if label is None:
            continue
        expected = label.upper()
        raw = result.raw_status.name
        effective = result.effective_status.name
        confusion[expected][raw] += 1
        effective_confusion[expected][effective] += 1
        labelled += 1
        raw_correct += int(raw == expected)
        effective_correct += int(effective == expected)
        if previous_raw is not None and raw != previous_raw:
            raw_changes += 1
        if previous_effective is not None and effective != previous_effective:
            effective_changes += 1
        previous_raw = raw
        previous_effective = effective

    labels = ["NORMAL", "WARNING", "CRITICAL"]
    print(f"Rows evaluated: {labelled}")
    print(f"Raw manufacturer-rule accuracy: {raw_correct / labelled:.4%}")
    print(f"Stabilized operational accuracy: {effective_correct / labelled:.4%}")
    print(f"Raw status changes: {raw_changes}")
    print(f"Stabilized status changes: {effective_changes}")
    print("\nRaw confusion matrix (expected rows, predicted columns):")
    print("expected\\pred " + " ".join(f"{label:>10}" for label in labels))
    for expected in labels:
        print(f"{expected:>13} " + " ".join(f"{confusion[expected][predicted]:10d}" for predicted in labels))
    print("\nStabilized confusion matrix:")
    print("expected\\pred " + " ".join(f"{label:>10}" for label in labels))
    for expected in labels:
        print(f"{expected:>13} " + " ".join(f"{effective_confusion[expected][predicted]:10d}" for predicted in labels))


if __name__ == "__main__":
    main()
