#!/usr/bin/env python3
r"""
Evaluate false positives on the spindle negative-only stress-test dataset.

Every source row is ground-truth NORMAL, so:
    false-positive rate = predicted WARNING/CRITICAL rows / analyzed rows

Example:
    python evaluate_fp_rate.py ^
        --replay output\fp_test\replay.csv ^
        --manifest data\spindle_fp_rate_stress_test_manifest.csv

If auto-detection selects the wrong output column:
    python evaluate_fp_rate.py --replay ... --prediction-column final_status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PREDICTION_CANDIDATES = (
    "predicted_status",
    "final_status",
    "model_status",
    "output_status",
    "smoothed_status",
    "alert_status",
    "status",
    "prediction",
)

TIMESTAMP_CANDIDATES = (
    "timestamp",
    "event_timestamp",
    "reading_timestamp",
    "time",
)


def detect_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    exact = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
    return None


def normalize_status(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True, help="Replay output CSV")
    parser.add_argument(
        "--prediction-column",
        help="Column containing the model's final predicted status",
    )
    parser.add_argument(
        "--timestamp-column",
        help="Timestamp column; auto-detected when omitted",
    )
    parser.add_argument(
        "--manifest",
        help="Optional scenario manifest CSV for scenario-level FP rates",
    )
    args = parser.parse_args()

    replay_path = Path(args.replay)
    if not replay_path.exists():
        print(f"ERROR: replay file not found: {replay_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(replay_path)
    if df.empty:
        print("ERROR: replay output contains no rows.", file=sys.stderr)
        return 2

    prediction_column = args.prediction_column or detect_column(
        list(df.columns), PREDICTION_CANDIDATES
    )
    if prediction_column is None:
        print("ERROR: Could not identify the prediction-status column.")
        print("Available columns:")
        for column in df.columns:
            print(f"  - {column}")
        print("\nRun again with --prediction-column COLUMN_NAME.")
        return 2

    statuses = normalize_status(df[prediction_column])
    valid_statuses = {"NORMAL", "WARNING", "CRITICAL"}
    unknown = sorted(set(statuses.dropna()) - valid_statuses)
    if unknown:
        print(
            f"WARNING: Unrecognized values in {prediction_column}: {unknown[:10]}"
        )

    is_fp = statuses.isin({"WARNING", "CRITICAL"})
    total = len(df)
    fp_rows = int(is_fp.sum())
    tn_rows = total - fp_rows
    warning_rows = int((statuses == "WARNING").sum())
    critical_rows = int((statuses == "CRITICAL").sum())
    fp_rate = fp_rows / total if total else float("nan")

    episode_starts = is_fp & ~is_fp.shift(fill_value=False)
    episode_ids = episode_starts.cumsum()
    false_rows = df.loc[is_fp].copy()
    false_rows["_episode_id"] = episode_ids[is_fp].to_numpy()
    false_rows["_status_normalized"] = statuses[is_fp].to_numpy()

    episode_count = int(episode_starts.sum())
    longest_episode_rows = 0
    if not false_rows.empty:
        longest_episode_rows = int(
            false_rows.groupby("_episode_id").size().max()
        )

    print("\nFALSE-POSITIVE EVALUATION")
    print("=" * 52)
    print(f"Replay file             : {replay_path}")
    print(f"Prediction column       : {prediction_column}")
    print(f"Rows analyzed           : {total:,}")
    print(f"True negatives          : {tn_rows:,}")
    print(f"False-positive rows     : {fp_rows:,}")
    print(f"  WARNING               : {warning_rows:,}")
    print(f"  CRITICAL              : {critical_rows:,}")
    print(f"Row-level FP rate       : {fp_rate:.6%}")
    print(f"False-alert episodes    : {episode_count:,}")
    print(f"Longest FP episode      : {longest_episode_rows:,} rows")

    if total != 100_000:
        print(
            "\nWARNING: This test dataset should produce 100,000 analyzed rows. "
            "Check invalid_rows.csv or replay filtering before accepting the score."
        )

    timestamp_column = args.timestamp_column or detect_column(
        list(df.columns), TIMESTAMP_CANDIDATES
    )

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"\nWARNING: manifest not found: {manifest_path}")
        elif timestamp_column is None:
            print(
                "\nWARNING: no timestamp column detected; "
                "scenario-level evaluation was skipped."
            )
        else:
            manifest = pd.read_csv(manifest_path)
            df["_evaluation_timestamp"] = pd.to_datetime(
                df[timestamp_column], errors="coerce"
            )
            df["_is_fp"] = is_fp.to_numpy()

            print("\nFP RATE BY TEST SCENARIO")
            print("=" * 88)
            print(
                f"{'Scenario':36} {'Rows':>10} {'FP rows':>10} "
                f"{'FP rate':>14} {'Episodes':>10}"
            )
            print("-" * 88)

            for _, row in manifest.iterrows():
                start = pd.to_datetime(row["start_timestamp"])
                end = pd.to_datetime(row["end_timestamp"])
                subset = df[
                    (df["_evaluation_timestamp"] >= start)
                    & (df["_evaluation_timestamp"] <= end)
                ]
                subset_fp = subset["_is_fp"].fillna(False).astype(bool)
                subset_count = len(subset)
                subset_fp_count = int(subset_fp.sum())
                subset_rate = (
                    subset_fp_count / subset_count if subset_count else float("nan")
                )
                subset_episodes = int(
                    (subset_fp & ~subset_fp.shift(fill_value=False)).sum()
                )
                print(
                    f"{str(row['scenario']):36} "
                    f"{subset_count:10,d} "
                    f"{subset_fp_count:10,d} "
                    f"{subset_rate:13.6%} "
                    f"{subset_episodes:10,d}"
                )

    if fp_rows == 0:
        print("\nRESULT: Perfect on this test: 0 false-positive rows.")
    elif fp_rate <= 0.001:
        print("\nRESULT: Very low row-level FP rate on this synthetic stress test.")
    elif fp_rate <= 0.01:
        print("\nRESULT: Non-trivial FP rate; inspect episodes and difficult scenarios.")
    else:
        print("\nRESULT: High FP rate; the model/guardrail needs investigation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
