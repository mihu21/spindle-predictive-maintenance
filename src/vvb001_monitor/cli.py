from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from .checkpoint import CheckpointStore
from .config import AppConfig
from .features import FeatureEngine
from .generalization import GeneralizationCriteria, evaluate_generalization
from .monitor import VVB001Monitor
from .plant_shadow.commands import add_plant_shadow_parser, run_plant_shadow_command
from .postgres_source import PostgresVVB001Source
from .predictor import VVB001Predictor
from .rul_evaluation import RULEvaluationCriteria, evaluate_rul
from .rul_v2_2 import generate_rul_v2_2_development_corpus, train_rul_v2_2_model
from .rul_v2_3 import generate_rul_v2_3_development_corpus, train_rul_v2_3_model
from .rul_v2_4 import (
    evaluate_rul_v2_4_development_acceptance,
    generate_rul_v2_4_development_corpus,
    train_rul_v2_4_candidate,
)
from .rul_v2_5 import (
    evaluate_rul_v2_5_development_acceptance,
    generate_rul_v2_5_development_corpus,
    train_rul_v2_5_candidate,
)
from .rul_v2_6 import (
    evaluate_rul_v2_6_development_acceptance,
    generate_rul_v2_6_acceptance_after_parity,
    generate_rul_v2_6_preacceptance_corpus,
    train_rul_v2_6_candidate,
)
from .rul_v2_7 import (
    evaluate_rul_v2_7_development_acceptance,
    generate_rul_v2_7_acceptance_after_parity,
    generate_rul_v2_7_preacceptance_corpus,
    train_rul_v2_7_candidate,
)
from .rul_v2_7_sealed import (
    evaluate_rul_v2_7_sealed_holdout,
    generate_rul_v2_7_sealed_holdout,
)
from .sensor_quality import SensorQualityGuard
from .storage import LocalStore, export_csv
from .stress import StressCriteria, StressSuiteConfig, evaluate_stress_suite, generate_stress_suite
from .synthetic import SyntheticConfig, generate_mock_csv
from .training import train_bootstrap_model
from .validation import VVB001Validator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IFM VVB001 spindle prognostics pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    mock = sub.add_parser(
        "generate-mock",
        help="Generate diverse unlabelled VVB001 degradation lifecycles for bootstrap learning",
    )
    mock.add_argument("--output", default="data/vvb001_mock_training.csv")
    mock.add_argument("--lifecycles", type=int, default=50)
    mock.add_argument("--machines", type=int, default=6)
    mock.add_argument("--line-sel", default="LINE_1")
    mock.add_argument("--cadence-seconds", type=int, default=60)
    mock.add_argument("--seed", type=int, default=42)
    mock.add_argument(
        "--duty-cycled-operating-context",
        action="store_true",
        help="Add OFF/IDLE/MAINTENANCE/UNKNOWN test rows; output is runtime-test-only, not training data",
    )
    mock.add_argument(
        "--vibration-inference-fixture",
        action="store_true",
        help="Blank state inputs while retaining hidden synthetic truth for detector-only testing",
    )

    train = sub.add_parser(
        "train-mock",
        help="Discover degradation regimes from synthetic lifecycles without status labels",
    )
    train.add_argument("--input", default="data/vvb001_mock_training.csv")
    train.add_argument("--config", default="config/vvb001.json")
    train.add_argument("--model", default="models/vvb001_bootstrap.joblib")
    train.add_argument("--report", default="output/bootstrap_training_report.json")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--baseline-anchor-fraction",
        type=float,
        default=0.15,
        help="Early lifecycle fraction used only to identify the learned healthy regime (default 0.15)",
    )

    rul_v22_generate = sub.add_parser(
        "generate-rul-v2-2-development",
        help="Generate the development-only multi-seed RUL v2.2 corpus and provenance manifest",
    )
    rul_v22_generate.add_argument("--output", default="data/rul_v2_2_development.csv")
    rul_v22_generate.add_argument("--manifest", default="output/rul_v2_2/corpus_manifest.json")
    rul_v22_generate.add_argument("--seeds", type=int, nargs="+", default=[2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208])
    rul_v22_generate.add_argument("--lifecycles-per-seed", type=int, default=20)
    rul_v22_generate.add_argument("--machines", type=int, default=6)
    rul_v22_generate.add_argument("--cadence-seconds", type=int, default=600)
    rul_v22_generate.add_argument("--line-sel", default="LINE_1")

    rul_v22_train = sub.add_parser(
        "train-rul-v2-2",
        help="Train RUL v2.2 on seed-disjoint development data while freezing status/safety behavior",
    )
    rul_v22_train.add_argument("--input", default="data/rul_v2_2_development.csv")
    rul_v22_train.add_argument("--manifest", default="output/rul_v2_2/corpus_manifest.json")
    rul_v22_train.add_argument("--frozen-status-model", default="models/model_10min.joblib")
    rul_v22_train.add_argument("--config", default="config/vvb001.json")
    rul_v22_train.add_argument("--model", default="models/rul_v2_2_development.joblib")
    rul_v22_train.add_argument("--report", default="output/rul_v2_2/development_training_report.json")
    rul_v22_train.add_argument("--candidate-report", default="output/rul_v2_2/candidate_comparison.json")
    rul_v22_train.add_argument("--horizon-report", default="output/rul_v2_2/horizon_diagnostics.json")
    rul_v22_train.add_argument("--seed", type=int, default=2200)

    rul_v23_generate = sub.add_parser(
        "generate-rul-v2-3-development",
        help="Generate fresh permanently role-locked fit/calibration/acceptance batches for RUL v2.3",
    )
    rul_v23_generate.add_argument("--output", default="data/rul_v2_3_development.csv")
    rul_v23_generate.add_argument("--manifest", default="output/rul_v2_3/corpus_manifest.json")
    rul_v23_generate.add_argument("--consumed-registry", default="output/rul_v2_3/consumed_evidence_registry.json")
    rul_v23_generate.add_argument("--fit-seeds", type=int, nargs="+", default=[23001, 23002, 23003, 23004, 23005, 23006, 23007, 23008])
    rul_v23_generate.add_argument("--calibration-seeds", type=int, nargs="+", default=[23101, 23102, 23103, 23104])
    rul_v23_generate.add_argument("--acceptance-seeds", type=int, nargs="+", default=[23201, 23202, 23203, 23204])
    rul_v23_generate.add_argument("--lifecycles-per-batch", type=int, default=12)
    rul_v23_generate.add_argument("--machines", type=int, default=6)
    rul_v23_generate.add_argument("--cadence-seconds", type=int, default=600)
    rul_v23_generate.add_argument("--line-sel", default="LINE_1")

    rul_v23_train = sub.add_parser(
        "train-rul-v2-3",
        help="Run forecastability audit and train the fail-closed RUL v2.3 development contract",
    )
    rul_v23_train.add_argument("--input", default="data/rul_v2_3_development.csv")
    rul_v23_train.add_argument("--manifest", default="output/rul_v2_3/corpus_manifest.json")
    rul_v23_train.add_argument("--consumed-registry", default="output/rul_v2_3/consumed_evidence_registry.json")
    rul_v23_train.add_argument("--frozen-status-model", default="models/model_10min.joblib")
    rul_v23_train.add_argument("--config", default="config/vvb001.json")
    rul_v23_train.add_argument("--model", default="models/rul_v2_3_development.joblib")
    rul_v23_train.add_argument("--output-dir", default="output/rul_v2_3")
    rul_v23_train.add_argument("--seed", type=int, default=2300)

    rul_v24_generate = sub.add_parser(
        "generate-rul-v2-4-development",
        help="Generate fresh physically separated RUL v2.4 fit/calibration/acceptance evidence",
    )
    rul_v24_generate.add_argument("--data-dir", default="data/rul_v2_4")
    rul_v24_generate.add_argument("--manifest", default="output/rul_v2_4/corpus_manifest.json")
    rul_v24_generate.add_argument("--consumed-registry", default="output/rul_v2_4/consumed_evidence_registry.json")
    rul_v24_generate.add_argument("--fit-seeds", type=int, nargs="+", default=[24001, 24002, 24003, 24004, 24005, 24006, 24007, 24008])
    rul_v24_generate.add_argument("--calibration-seeds", type=int, nargs="+", default=[24101, 24102, 24103, 24104])
    rul_v24_generate.add_argument("--acceptance-seeds", type=int, nargs="+", default=[24201, 24202, 24203, 24204])
    rul_v24_generate.add_argument("--lifecycles-per-batch", type=int, default=12)
    rul_v24_generate.add_argument("--machines", type=int, default=6)
    rul_v24_generate.add_argument("--cadence-seconds", type=int, default=600)
    rul_v24_generate.add_argument("--line-sel", default="LINE_1")

    rul_v24_train = sub.add_parser(
        "train-rul-v2-4",
        help="Freeze the nested-crossfit v2.4 candidate without opening acceptance",
    )
    rul_v24_train.add_argument("--manifest", default="output/rul_v2_4/corpus_manifest.json")
    rul_v24_train.add_argument("--consumed-registry", default="output/rul_v2_4/consumed_evidence_registry.json")
    rul_v24_train.add_argument("--freeze-manifest", default="output/rul_v2_4/preacceptance_freeze.json")
    rul_v24_train.add_argument("--frozen-status-model", default="models/model_10min.joblib")
    rul_v24_train.add_argument("--config", default="config/vvb001.json")
    rul_v24_train.add_argument("--model", default="models/rul_v2_4_development.joblib")
    rul_v24_train.add_argument("--output-dir", default="output/rul_v2_4")
    rul_v24_train.add_argument("--seed", type=int, default=2400)

    rul_v24_accept = sub.add_parser(
        "evaluate-rul-v2-4-development",
        help="Open and consume fresh v2.4 acceptance after verifying the frozen candidate",
    )
    rul_v24_accept.add_argument("--manifest", default="output/rul_v2_4/corpus_manifest.json")
    rul_v24_accept.add_argument("--consumed-registry", default="output/rul_v2_4/consumed_evidence_registry.json")
    rul_v24_accept.add_argument("--freeze-manifest", default="output/rul_v2_4/preacceptance_freeze.json")
    rul_v24_accept.add_argument("--model", default="models/rul_v2_4_development.joblib")
    rul_v24_accept.add_argument("--frozen-v2-3-model", default="models/rul_v2_3_development.joblib")
    rul_v24_accept.add_argument("--config", default="config/vvb001.json")
    rul_v24_accept.add_argument("--output-dir", default="output/rul_v2_4")

    rul_v25_generate = sub.add_parser(
        "generate-rul-v2-5-development",
        help="Generate fresh role-locked v2.5 fit/calibration/acceptance evidence",
    )
    rul_v25_generate.add_argument("--data-dir", default="data/rul_v2_5")
    rul_v25_generate.add_argument("--manifest", default="output/rul_v2_5/corpus_manifest.json")
    rul_v25_generate.add_argument("--consumed-registry", default="output/rul_v2_5/consumed_evidence_registry.json")
    rul_v25_generate.add_argument("--fit-seeds", type=int, nargs="+", default=[25001, 25002, 25003, 25004, 25005, 25006, 25007, 25008])
    rul_v25_generate.add_argument("--calibration-seeds", type=int, nargs="+", default=[25101, 25102, 25103, 25104])
    rul_v25_generate.add_argument("--acceptance-seeds", type=int, nargs="+", default=[25201, 25202, 25203, 25204])
    rul_v25_generate.add_argument("--lifecycles-per-batch", type=int, default=12)
    rul_v25_generate.add_argument("--machines", type=int, default=6)
    rul_v25_generate.add_argument("--cadence-seconds", type=int, default=600)
    rul_v25_generate.add_argument("--line-sel", default="LINE_1")

    rul_v25_train = sub.add_parser(
        "train-rul-v2-5",
        help="Freeze v2.5 final-active calibration without opening acceptance",
    )
    rul_v25_train.add_argument("--manifest", default="output/rul_v2_5/corpus_manifest.json")
    rul_v25_train.add_argument("--consumed-registry", default="output/rul_v2_5/consumed_evidence_registry.json")
    rul_v25_train.add_argument("--freeze-manifest", default="output/rul_v2_5/preacceptance_freeze_v2_5.json")
    rul_v25_train.add_argument("--frozen-v2-4-model", default="models/rul_v2_4_development.joblib")
    rul_v25_train.add_argument("--config", default="config/vvb001.json")
    rul_v25_train.add_argument("--model", default="models/rul_v2_5_development.joblib")
    rul_v25_train.add_argument("--output-dir", default="output/rul_v2_5")

    rul_v25_accept = sub.add_parser(
        "evaluate-rul-v2-5-development",
        help="Open and consume fresh v2.5 acceptance after verifying the full freeze",
    )
    rul_v25_accept.add_argument("--manifest", default="output/rul_v2_5/corpus_manifest.json")
    rul_v25_accept.add_argument("--consumed-registry", default="output/rul_v2_5/consumed_evidence_registry.json")
    rul_v25_accept.add_argument("--freeze-manifest", default="output/rul_v2_5/preacceptance_freeze_v2_5.json")
    rul_v25_accept.add_argument("--model", default="models/rul_v2_5_development.joblib")
    rul_v25_accept.add_argument("--frozen-v2-4-model", default="models/rul_v2_4_development.joblib")
    rul_v25_accept.add_argument("--config", default="config/vvb001.json")
    rul_v25_accept.add_argument("--output-dir", default="output/rul_v2_5")

    rul_v26_generate = sub.add_parser(
        "generate-rul-v2-6-preacceptance",
        help="Generate only fresh v2.6 fit/calibration evidence and predeclare acceptance seeds",
    )
    rul_v26_generate.add_argument("--data-dir", default="data/rul_v2_6")
    rul_v26_generate.add_argument("--manifest", default="output/rul_v2_6/corpus_manifest.json")
    rul_v26_generate.add_argument("--consumed-registry", default="output/rul_v2_6/consumed_evidence_registry.json")
    rul_v26_generate.add_argument("--fit-seeds", type=int, nargs="+", default=[26001, 26002, 26003, 26004, 26005, 26006, 26007, 26008])
    rul_v26_generate.add_argument("--calibration-seeds", type=int, nargs="+", default=[26101, 26102, 26103, 26104])
    rul_v26_generate.add_argument("--acceptance-seeds", type=int, nargs="+", default=[26201, 26202, 26203, 26204])
    rul_v26_generate.add_argument("--lifecycles-per-batch", type=int, default=8)
    rul_v26_generate.add_argument("--machines", type=int, default=4)
    rul_v26_generate.add_argument("--cadence-seconds", type=int, default=600)
    rul_v26_generate.add_argument("--line-sel", default="LINE_1")

    rul_v26_train = sub.add_parser(
        "train-rul-v2-6", help="Fit target-specific correction/calibration and require reload parity",
    )
    rul_v26_train.add_argument("--manifest", default="output/rul_v2_6/corpus_manifest.json")
    rul_v26_train.add_argument("--consumed-registry", default="output/rul_v2_6/consumed_evidence_registry.json")
    rul_v26_train.add_argument("--freeze-manifest", default="output/rul_v2_6/preacceptance_freeze_v2_6.json")
    rul_v26_train.add_argument("--frozen-v2-5-model", default="models/rul_v2_5_development.joblib")
    rul_v26_train.add_argument("--config", default="config/vvb001.json")
    rul_v26_train.add_argument("--model", default="models/rul_v2_6_development.joblib")
    rul_v26_train.add_argument("--output-dir", default="output/rul_v2_6")

    rul_v26_accept_generate = sub.add_parser(
        "generate-rul-v2-6-acceptance",
        help="Generate predeclared v2.6 acceptance only after final-artifact parity passes",
    )
    rul_v26_accept_generate.add_argument("--manifest", default="output/rul_v2_6/corpus_manifest.json")
    rul_v26_accept_generate.add_argument("--consumed-registry", default="output/rul_v2_6/consumed_evidence_registry.json")
    rul_v26_accept_generate.add_argument("--freeze-manifest", default="output/rul_v2_6/preacceptance_freeze_v2_6.json")

    rul_v26_accept = sub.add_parser(
        "evaluate-rul-v2-6-development", help="Open and consume fresh v2.6 acceptance exactly once",
    )
    rul_v26_accept.add_argument("--manifest", default="output/rul_v2_6/corpus_manifest.json")
    rul_v26_accept.add_argument("--consumed-registry", default="output/rul_v2_6/consumed_evidence_registry.json")
    rul_v26_accept.add_argument("--freeze-manifest", default="output/rul_v2_6/preacceptance_freeze_v2_6.json")
    rul_v26_accept.add_argument("--model", default="models/rul_v2_6_development.joblib")
    rul_v26_accept.add_argument("--config", default="config/vvb001.json")
    rul_v26_accept.add_argument("--output-dir", default="output/rul_v2_6")

    rul_v27_generate = sub.add_parser(
        "generate-rul-v2-7-preacceptance",
        help="Generate fresh identity-first v2.7 fit/calibration evidence and predeclare acceptance",
    )
    rul_v27_generate.add_argument("--data-dir", default="data/rul_v2_7")
    rul_v27_generate.add_argument("--manifest", default="output/rul_v2_7/corpus_manifest.json")
    rul_v27_generate.add_argument("--consumed-registry", default="output/rul_v2_7/consumed_evidence_registry.json")
    rul_v27_generate.add_argument("--fit-seeds", type=int, nargs="+", default=[28001, 28002, 28003, 28004, 28005, 28006, 28007, 28008])
    rul_v27_generate.add_argument("--calibration-seeds", type=int, nargs="+", default=[28101, 28102, 28103, 28104])
    rul_v27_generate.add_argument("--acceptance-seeds", type=int, nargs="+", default=[28201, 28202, 28203, 28204])
    rul_v27_generate.add_argument("--lifecycles-per-batch", type=int, default=8)
    rul_v27_generate.add_argument("--machines", type=int, default=4)
    rul_v27_generate.add_argument("--cadence-seconds", type=int, default=600)
    rul_v27_generate.add_argument("--line-sel", default="LINE_1")

    rul_v27_train = sub.add_parser(
        "train-rul-v2-7",
        help="Select CRITICAL identity vs one constrained correction and require runtime parity",
    )
    rul_v27_train.add_argument("--manifest", default="output/rul_v2_7/corpus_manifest.json")
    rul_v27_train.add_argument("--consumed-registry", default="output/rul_v2_7/consumed_evidence_registry.json")
    rul_v27_train.add_argument("--freeze-manifest", default="output/rul_v2_7/preacceptance_freeze_v2_7.json")
    rul_v27_train.add_argument("--frozen-v2-6-model", default="models/rul_v2_6_full_cadence.joblib")
    rul_v27_train.add_argument("--config", default="config/vvb001.json")
    rul_v27_train.add_argument("--model", default="models/rul_v2_7_development.joblib")
    rul_v27_train.add_argument("--output-dir", default="output/rul_v2_7")

    rul_v27_accept_generate = sub.add_parser(
        "generate-rul-v2-7-acceptance",
        help="Generate predeclared v2.7 acceptance only after every preacceptance gate passes",
    )
    rul_v27_accept_generate.add_argument("--manifest", default="output/rul_v2_7/corpus_manifest.json")
    rul_v27_accept_generate.add_argument("--consumed-registry", default="output/rul_v2_7/consumed_evidence_registry.json")
    rul_v27_accept_generate.add_argument("--freeze-manifest", default="output/rul_v2_7/preacceptance_freeze_v2_7.json")

    rul_v27_accept = sub.add_parser(
        "evaluate-rul-v2-7-development",
        help="Open and consume fresh v2.7 acceptance exactly once",
    )
    rul_v27_accept.add_argument("--manifest", default="output/rul_v2_7/corpus_manifest.json")
    rul_v27_accept.add_argument("--consumed-registry", default="output/rul_v2_7/consumed_evidence_registry.json")
    rul_v27_accept.add_argument("--freeze-manifest", default="output/rul_v2_7/preacceptance_freeze_v2_7.json")
    rul_v27_accept.add_argument("--model", default="models/rul_v2_7_development.joblib")
    rul_v27_accept.add_argument("--config", default="config/vvb001.json")
    rul_v27_accept.add_argument("--output-dir", default="output/rul_v2_7")

    rul_v27_sealed_generate = sub.add_parser(
        "generate-rul-v2-7-sealed-holdout",
        help="Generate the authorized one-time 8-batch v2.7 sealed synthetic holdout",
    )
    rul_v27_sealed_generate.add_argument("--data-dir", default="data/rul_v2_7_sealed")
    rul_v27_sealed_generate.add_argument("--manifest", default="output/rul_v2_7_sealed/sealed_holdout_manifest.json")
    rul_v27_sealed_generate.add_argument("--consumed-registry", default="output/rul_v2_7_full_cadence/consumed_evidence_registry.json")
    rul_v27_sealed_generate.add_argument("--acceptance-freeze", default="output/rul_v2_7_full_cadence/preacceptance_freeze_v2_7.json")
    rul_v27_sealed_generate.add_argument("--model", default="models/rul_v2_7_full_cadence.joblib")
    rul_v27_sealed_generate.add_argument("--seeds", type=int, nargs="+", default=[28301, 28302, 28303, 28304, 28305, 28306, 28307, 28308])
    rul_v27_sealed_generate.add_argument("--lifecycles-per-batch", type=int, default=8)
    rul_v27_sealed_generate.add_argument("--machines", type=int, default=4)
    rul_v27_sealed_generate.add_argument("--cadence-seconds", type=int, default=600)
    rul_v27_sealed_generate.add_argument("--line-sel", default="LINE_1")

    rul_v27_sealed_evaluate = sub.add_parser(
        "evaluate-rul-v2-7-sealed-holdout",
        help="Open and consume the frozen v2.7 sealed synthetic holdout exactly once",
    )
    rul_v27_sealed_evaluate.add_argument("--manifest", default="output/rul_v2_7_sealed/sealed_holdout_manifest.json")
    rul_v27_sealed_evaluate.add_argument("--consumed-registry", default="output/rul_v2_7_full_cadence/consumed_evidence_registry.json")
    rul_v27_sealed_evaluate.add_argument("--model", default="models/rul_v2_7_full_cadence.joblib")
    rul_v27_sealed_evaluate.add_argument("--config", default="config/vvb001.json")
    rul_v27_sealed_evaluate.add_argument("--output-dir", default="output/rul_v2_7_sealed")

    generalization = sub.add_parser(
        "evaluate-generalization",
        help="Test a fixed trained model on fresh unseen synthetic lifecycles without retraining",
    )
    generalization.add_argument("--model", default="models/vvb001_bootstrap.joblib")
    generalization.add_argument("--config", default="config/vvb001.json")
    generalization.add_argument("--report", default="output/generalization_evaluation.json")
    generalization.add_argument("--trials", type=int, default=5)
    generalization.add_argument("--lifecycles", type=int, default=30)
    generalization.add_argument("--machines", type=int, default=6)
    generalization.add_argument(
        "--cadence-seconds",
        type=int,
        default=None,
        help="Synthetic evaluation cadence; defaults to cadence stored in newly trained models, otherwise 60",
    )
    generalization.add_argument("--seed-start", type=int, default=1001)
    generalization.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit unique evaluation seeds; when supplied, overrides --trials/--seed-start",
    )
    generalization.add_argument("--line-sel", default="LINE_1")
    generalization.add_argument("--keep-generated-dir", default=None)
    generalization.add_argument("--max-early-alert-rate", type=float, default=0.10)
    generalization.add_argument("--min-late-critical-coverage", type=float, default=0.20)
    generalization.add_argument("--min-progress-spearman", type=float, default=0.65)
    generalization.add_argument("--min-hidden-damage-spearman", type=float, default=0.70)
    generalization.add_argument("--max-transition-reversal-rate", type=float, default=0.20)

    rul_eval = sub.add_parser(
        "evaluate-rul",
        help="Replay synthetic lifecycles causally and evaluate estimated hours-to-WARNING/CRITICAL against hidden future truth",
    )
    rul_eval.add_argument("--model", default="models/vvb001_bootstrap.joblib")
    rul_eval.add_argument("--config", default="config/vvb001.json")
    rul_eval.add_argument("--input", default="data/vvb001_mock_training.csv")
    rul_eval.add_argument("--ground-truth", default=None, help="Optional separate hidden-truth CSV, e.g. a stress-suite ground-truth file")
    rul_eval.add_argument("--report", default="output/rul_evaluation.json")
    rul_eval.add_argument("--predictions", default="output/rul_predictions.csv")
    rul_eval.add_argument("--max-critical-mae-hours", type=float, default=12.0)
    rul_eval.add_argument("--max-critical-median-abs-error-hours", type=float, default=10.0)
    rul_eval.add_argument("--min-critical-estimate-availability", type=float, default=0.95)
    rul_eval.add_argument("--min-rul-monotonicity", type=float, default=0.95)
    rul_eval.add_argument("--min-interval-coverage", type=float, default=0.80)
    rul_eval.add_argument("--min-macro-lifecycle-interval-coverage", type=float, default=0.75)
    rul_eval.add_argument("--max-within-24h-mae-hours", type=float, default=6.0)
    rul_eval.add_argument("--max-within-12h-mae-hours", type=float, default=5.0)
    rul_eval.add_argument("--max-within-6h-mae-hours", type=float, default=4.0)
    rul_eval.add_argument("--max-premature-short-rul-rate", type=float, default=0.10)
    rul_eval.add_argument("--max-sensor-fault-large-rul-jump-rate", type=float, default=0.25)

    stress_generate = sub.add_parser(
        "generate-stress",
        help="Generate a frozen versioned independent stress suite; ground truth is stored separately from sensor data",
    )
    stress_generate.add_argument("--output-dir", default="data/stress")
    stress_generate.add_argument(
        "--suite-version",
        choices=("stress_v1", "stress_v2", "stress_v3", "stress_v4"),
        default="stress_v4",
    )
    stress_generate.add_argument("--cadence-seconds", type=int, default=600)
    stress_generate.add_argument("--replicates", type=int, default=2)
    stress_generate.add_argument("--machines", type=int, default=6)
    stress_generate.add_argument("--line-sel", default="LINE_1")
    stress_generate.add_argument("--overwrite", action="store_true")

    stress = sub.add_parser(
        "evaluate-stress",
        help="Evaluate a fixed model against a frozen versioned stress suite without retraining or calibration",
    )
    stress.add_argument("--model", default="models/vvb001_bootstrap.joblib")
    stress.add_argument("--config", default="config/vvb001.json")
    stress.add_argument("--suite-dir", default="data/stress/stress_v1")
    stress.add_argument("--report", default="output/stress/stress_v1_evaluation.json")
    stress.add_argument("--max-overall-fp-rate", type=float, default=0.15)
    stress.add_argument("--max-overall-fn-rate", type=float, default=0.35)
    stress.add_argument("--min-overall-critical-recall", type=float, default=0.35)
    stress.add_argument("--max-healthy-alert-rate", type=float, default=0.20)
    stress.add_argument("--max-sensor-fault-alert-rate", type=float, default=0.25)
    stress.add_argument("--max-fault-fn-rate", type=float, default=0.40)
    stress.add_argument("--min-fault-critical-recall", type=float, default=0.30)
    stress.add_argument("--min-sudden-critical-recall", type=float, default=0.50)

    live = sub.add_parser("monitor-postgres", help="Read VVB001 rows from PostgreSQL without modifying PostgreSQL")
    live.add_argument("--config", default="config/vvb001.json")
    live.add_argument("--model", default="models/vvb001_bootstrap.joblib")
    live.add_argument("--collect-only", action="store_true", help="Collect/feature-engineer data without loading a model")
    live.add_argument("--database", default="output/vvb001_monitor.db")
    live.add_argument("--checkpoint", default="output/vvb001_checkpoint.json")
    live.add_argument("--once", action="store_true", help="Catch up currently available rows then exit")
    live.add_argument("--start-from-beginning", action="store_true")
    live.add_argument("--poll-seconds", type=float, default=None)
    live.add_argument("--batch-size", type=int, default=None)
    live.add_argument("--progress-every", type=int, default=1000)

    export = sub.add_parser("export-csv", help="Export local readings, predictions and features to CSV")
    export.add_argument("--database", default="output/vvb001_monitor.db")
    export.add_argument("--output", default="output/vvb001_training_features.csv")

    profile = sub.add_parser("profile-local", help="Print aggregate and per-machine profiles of locally collected VVB001 readings")
    profile.add_argument("--database", default="output/vvb001_monitor.db")

    latest = sub.add_parser("show-latest", help="Show the latest status and RUL estimate for each monitored machine")
    latest.add_argument("--database", default="output/vvb001_monitor.db")
    add_plant_shadow_parser(sub)
    return parser


def _process_records(
    records,
    monitor,
    local_store,
    checkpoint,
    checkpoint_store,
    *,
    persist: bool,
    progress_every: int,
    count_start: int = 0,
):
    count = count_start
    invalid = 0
    for record in records:
        validation, processed = monitor.process(record)
        if processed is None:
            invalid += 1
            if persist:
                local_store.save_invalid(validation, record)
        elif persist:
            local_store.save_processed(processed)
        if record.source_id >= 0:
            checkpoint.last_source_id = record.source_id
            if record.reading is not None:
                checkpoint.last_timestamp = record.reading.timestamp
            if persist:
                checkpoint.baseline_state = monitor.features.baseline_snapshot()
                checkpoint_store.save(checkpoint)
        count += 1
        if persist and progress_every > 0 and count % progress_every == 0:
            if processed is not None and processed.predicted_status is not None:
                crit = (
                    f"{processed.estimated_hours_to_critical:.1f}h"
                    if processed.estimated_hours_to_critical is not None
                    else "unavailable"
                )
                print(
                    f"Processed {count} PostgreSQL row(s); last source id={checkpoint.last_source_id}; "
                    f"status={processed.predicted_status}; score={processed.degradation_score:.4f}; "
                    f"RUL-to-CRITICAL={crit}; reliability={processed.rul_reliability}"
                )
            else:
                print(f"Processed {count} PostgreSQL row(s); last source id={checkpoint.last_source_id}")
    return count, invalid


def run_monitor_postgres(args: argparse.Namespace) -> None:
    config = AppConfig.load(args.config)
    pg = config.postgres
    if args.poll_seconds is not None:
        if args.poll_seconds < 0:
            raise ValueError("--poll-seconds cannot be negative")
        pg = replace(pg, poll_seconds=args.poll_seconds)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        pg = replace(pg, batch_size=args.batch_size)
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative")

    predictor = None
    if not args.collect_only:
        if not Path(args.model).exists():
            raise FileNotFoundError(
                f"Model not found: {args.model}. Run 'python main.py generate-mock' then 'python main.py train-mock', "
                "or use --collect-only."
            )
        predictor = VVB001Predictor(args.model, config.sensor)
        if predictor.metadata.get("bootstrap_only"):
            print("WARNING: loaded model learned regimes from synthetic lifecycles; live predictions are provisional.")

    checkpoint_store = CheckpointStore(args.checkpoint, pg.source_identity)
    checkpoint = checkpoint_store.load()
    if args.start_from_beginning and checkpoint.last_source_id is not None:
        raise RuntimeError("--start-from-beginning is only valid when no local checkpoint exists")

    source = PostgresVVB001Source(pg)
    store = LocalStore(args.database)
    feature_engine = FeatureEngine(config.sensor)
    feature_engine.restore_baselines(checkpoint.baseline_state)
    monitor = VVB001Monitor(VVB001Validator(config.sensor), feature_engine, predictor, SensorQualityGuard(config.sensor))
    total = invalid = 0
    try:
        source.connect()
        if checkpoint.last_source_id is not None and checkpoint.last_timestamp is not None:
            # Rebuild rolling/EWMA state causally. Frozen per-machine baselines come from the
            # local checkpoint, so restarts do not redefine a machine's healthy reference.
            start = checkpoint.last_timestamp - timedelta(hours=config.sensor.history_hours)
            warm = source.fetch_history_before(checkpoint.last_source_id, start)
            _process_records(warm, monitor, store, checkpoint, checkpoint_store, persist=False, progress_every=0)
        elif checkpoint.last_source_id is None and not args.start_from_beginning:
            recent = source.fetch_recent(config.sensor.history_hours)
            total, inv = _process_records(
                recent, monitor, store, checkpoint, checkpoint_store,
                persist=True, progress_every=args.progress_every, count_start=total,
            )
            invalid += inv
        while True:
            batch = source.fetch_after(checkpoint.last_source_id, pg.batch_size)
            if not batch:
                if args.once:
                    break
                time.sleep(pg.poll_seconds)
                continue
            total, inv = _process_records(
                batch, monitor, store, checkpoint, checkpoint_store,
                persist=True, progress_every=args.progress_every, count_start=total,
            )
            invalid += inv
    finally:
        source.close()
        store.close()
    print(f"Completed: {total} source row(s) processed this run; {invalid} invalid row(s).")
    print(f"Local SQLite: {Path(args.database).resolve()}")
    print(f"Local checkpoint: {Path(args.checkpoint).resolve()}")


def run_profile_local(args: argparse.Namespace) -> None:
    db = sqlite3.connect(args.database)
    row = db.execute(
        "SELECT COUNT(*), MIN(timestamp), MAX(timestamp), COUNT(DISTINCT line_sel || '::' || machine_id) FROM readings"
    ).fetchone()
    print(f"Rows: {row[0]}")
    print(f"Time range: {row[1]} -> {row[2]}")
    print(f"Machine streams: {row[3]}")
    print("\nAggregate:")
    for metric in ("vrms", "arms", "apeak", "crest", "temp"):
        stats = db.execute(f"SELECT MIN({metric}), AVG({metric}), MAX({metric}) FROM readings").fetchone()
        print(f"  {metric}: min={stats[0]}, mean={stats[1]}, max={stats[2]}")
    warnings = db.execute("SELECT COUNT(*) FROM readings WHERE quality_status='VALID_WITH_WARNING'").fetchone()[0]
    invalid = db.execute("SELECT COUNT(*) FROM invalid_rows").fetchone()[0]
    print(f"Quality warnings: {warnings}; invalid rows: {invalid}")

    machines = db.execute(
        "SELECT line_sel, machine_id, COUNT(*) FROM readings GROUP BY line_sel, machine_id ORDER BY line_sel, machine_id"
    ).fetchall()
    for line_sel, machine_id, count in machines:
        print(f"\n{line_sel}::{machine_id} ({count} rows)")
        for metric in ("vrms", "arms", "apeak", "crest", "temp"):
            stats = db.execute(
                f"SELECT MIN({metric}), AVG({metric}), MAX({metric}) FROM readings WHERE line_sel=? AND machine_id=?",
                (line_sel, machine_id),
            ).fetchone()
            print(f"  {metric}: min={stats[0]}, mean={stats[1]}, max={stats[2]}")
        pred_rows = db.execute(
            "SELECT predicted_status, COUNT(*) FROM readings WHERE line_sel=? AND machine_id=? "
            "AND predicted_status IS NOT NULL GROUP BY predicted_status ORDER BY predicted_status",
            (line_sel, machine_id),
        ).fetchall()
        if pred_rows:
            print("  predictions: " + ", ".join(f"{status}={n}" for status, n in pred_rows))
        score = db.execute(
            "SELECT AVG(degradation_score), MAX(degradation_score) FROM readings WHERE line_sel=? AND machine_id=? "
            "AND degradation_score IS NOT NULL",
            (line_sel, machine_id),
        ).fetchone()
        if score and score[0] is not None:
            print(f"  degradation_score: mean={score[0]:.4f}, max={score[1]:.4f}")
        latest = db.execute(
            "SELECT predicted_status, degradation_score, estimated_hours_to_warning, estimated_hours_to_critical, "
            "warning_rul_lower_hours, warning_rul_upper_hours, critical_rul_lower_hours, critical_rul_upper_hours, "
            "rul_reliability, rul_reason FROM readings WHERE line_sel=? AND machine_id=? ORDER BY timestamp DESC LIMIT 1",
            (line_sel, machine_id),
        ).fetchone()
        if latest and latest[0] is not None:
            print(
                f"  latest: status={latest[0]}, score={latest[1]:.4f}, "
                f"to_WARNING={_fmt_hours(latest[2])}, to_CRITICAL={_fmt_hours(latest[3])}, "
                f"RUL reliability={latest[8]}"
            )
            if latest[9]:
                print(f"  RUL reason: {latest[9]}")
    db.close()


def _fmt_hours(value: float | None) -> str:
    return "unavailable" if value is None else f"{float(value):.1f} h"


def run_show_latest(args: argparse.Namespace) -> None:
    db = sqlite3.connect(args.database)
    cols = {row[1] for row in db.execute("PRAGMA table_info(readings)")}
    if "estimated_hours_to_critical" not in cols:
        db.close()
        raise RuntimeError("This local database predates the RUL schema. Run monitor-postgres once with the updated code to migrate it.")
    rows = db.execute(
        """
        SELECT r.timestamp, r.line_sel, r.machine_id, r.predicted_status, r.degradation_score,
               r.estimated_hours_to_warning, r.warning_rul_lower_hours, r.warning_rul_upper_hours,
               r.estimated_hours_to_critical, r.critical_rul_lower_hours, r.critical_rul_upper_hours,
               r.rul_reliability, r.rul_reason, r.rul_state_source, r.sensor_quality_status
        FROM readings r
        JOIN (
            SELECT line_sel, machine_id, MAX(timestamp) AS max_timestamp
            FROM readings
            GROUP BY line_sel, machine_id
        ) latest
          ON latest.line_sel=r.line_sel AND latest.machine_id=r.machine_id AND latest.max_timestamp=r.timestamp
        ORDER BY r.line_sel, r.machine_id
        """
    ).fetchall()
    if not rows:
        print("No monitored readings found.")
        db.close()
        return
    for row in rows:
        (timestamp, line_sel, machine_id, status, score, warn, warn_lo, warn_hi, crit, crit_lo, crit_hi, reliability, reason, state_source, sensor_quality) = row
        print(f"{line_sel}::{machine_id} @ {timestamp}")
        print(f"  Status: {status or 'unavailable'}; degradation_score={score if score is not None else 'unavailable'}")
        print(f"  Estimated time to WARNING: {_fmt_hours(warn)}")
        if warn is not None and warn_lo is not None and warn_hi is not None:
            print(f"    range: {_fmt_hours(warn_lo)} - {_fmt_hours(warn_hi)}")
        print(f"  Estimated time to CRITICAL / RUL: {_fmt_hours(crit)}")
        if crit is not None and crit_lo is not None and crit_hi is not None:
            print(f"    range: {_fmt_hours(crit_lo)} - {_fmt_hours(crit_hi)}")
        print(f"  RUL reliability: {reliability}; state={state_source}; sensor_quality={sensor_quality}")
        if reason:
            print(f"  RUL reason: {reason}")
    db.close()


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate-mock":
        rows = generate_mock_csv(
            args.output,
            SyntheticConfig(
                lifecycles=args.lifecycles,
                cadence_seconds=args.cadence_seconds,
                seed=args.seed,
                machines=args.machines,
                line_sel=args.line_sel,
                duty_cycled_operating_context=args.duty_cycled_operating_context,
                vibration_inference_fixture=args.vibration_inference_fixture,
            ),
        )
        print(f"Generated {rows} synthetic VVB001 row(s): {Path(args.output).resolve()}")
        print("Training labels: none. latent_damage_score/fault_mode are simulator-only audit fields.")
        if args.duty_cycled_operating_context:
            print("Operating-context fixture: runtime-test-only; non-running rows are rejected by training.")
        if args.vibration_inference_fixture:
            print("Vibration-inference fixture: state inputs blanked; synthetic truth is audit-only.")
    elif args.command == "generate-rul-v2-2-development":
        manifest = generate_rul_v2_2_development_corpus(
            args.output,
            args.manifest,
            seeds=args.seeds,
            lifecycles_per_seed=args.lifecycles_per_seed,
            machines=args.machines,
            cadence_seconds=args.cadence_seconds,
            line_sel=args.line_sel,
        )
        print(f"RUL v2.2 development corpus: {Path(args.output).resolve()}")
        print(f"Manifest: {Path(args.manifest).resolve()}")
        print(f"Rows/lifecycles/seeds: {manifest['rows']}/{manifest['distribution']['lifecycle_count']}/{manifest['distribution']['seed_count']}")
        print("Protected external datasets were not used.")
    elif args.command == "train-rul-v2-2":
        config = AppConfig.load(args.config)
        report = train_rul_v2_2_model(
            args.input,
            args.manifest,
            args.frozen_status_model,
            args.model,
            args.report,
            args.candidate_report,
            args.horizon_report,
            config.sensor,
            seed=args.seed,
        )
        print(f"RUL v2.2 development model: {Path(args.model).resolve()}")
        print(f"Development report: {Path(args.report).resolve()}")
        print(f"Selected point estimator: {report['candidate_comparison']['selected']}")
        print(f"Development gates: {'PASS' if report['development_gates_passed'] else 'FAIL'}")
        print(f"Sealed holdout authorized: {report['sealed_holdout_authorized']}")
        print("WARNING: synthetic development model; not production ready.")
    elif args.command == "generate-rul-v2-3-development":
        manifest = generate_rul_v2_3_development_corpus(
            args.output,
            args.manifest,
            args.consumed_registry,
            role_seeds={
                "fit": args.fit_seeds,
                "calibration": args.calibration_seeds,
                "acceptance": args.acceptance_seeds,
            },
            lifecycles_per_batch=args.lifecycles_per_batch,
            machines=args.machines,
            cadence_seconds=args.cadence_seconds,
            line_sel=args.line_sel,
        )
        print(f"RUL v2.3 role-locked corpus: {Path(args.output).resolve()}")
        print(f"Manifest: {Path(args.manifest).resolve()}")
        for role, row in manifest["role_summary"].items():
            print(f"{role}: {row['batch_count']} batches, {row['lifecycle_count']} lifecycles, {row['rows']} rows")
        print("Protected and previously consumed evidence was not reused.")
    elif args.command == "train-rul-v2-3":
        config = AppConfig.load(args.config)
        report = train_rul_v2_3_model(
            args.input,
            args.manifest,
            args.consumed_registry,
            args.frozen_status_model,
            args.model,
            args.output_dir,
            config.sensor,
            seed=args.seed,
        )
        print(f"RUL v2.3 development model: {Path(args.model).resolve()}")
        print(f"Reports: {Path(args.output_dir).resolve()}")
        print("Forecastability: " + json.dumps(report["forecastability_classifications"], sort_keys=True))
        print(f"Development gates: {'PASS' if report['development_gates_passed'] else 'FAIL'}")
        print(f"Sealed holdout authorized: {report['sealed_holdout_authorized']}")
        print("WARNING: synthetic development evidence only; not production ready.")
    elif args.command == "generate-rul-v2-4-development":
        manifest = generate_rul_v2_4_development_corpus(
            args.data_dir,
            args.manifest,
            args.consumed_registry,
            role_seeds={
                "fit": args.fit_seeds,
                "calibration": args.calibration_seeds,
                "acceptance": args.acceptance_seeds,
            },
            lifecycles_per_batch=args.lifecycles_per_batch,
            machines=args.machines,
            cadence_seconds=args.cadence_seconds,
            line_sel=args.line_sel,
        )
        print(f"RUL v2.4 physically separated evidence: {Path(args.data_dir).resolve()}")
        for role, row in manifest["role_files"].items():
            print(f"{role}: {row['batches']} batches, {row['lifecycles']} lifecycles, {row['rows']} rows")
        print("Acceptance remains unopened; protected and consumed evidence was not reused.")
    elif args.command == "train-rul-v2-4":
        config = AppConfig.load(args.config)
        report = train_rul_v2_4_candidate(
            args.manifest,
            args.consumed_registry,
            args.frozen_status_model,
            args.model,
            args.output_dir,
            args.freeze_manifest,
            config.sensor,
            seed=args.seed,
        )
        print(f"Frozen RUL v2.4 candidate: {Path(args.model).resolve()}")
        print(f"Selected feature set: {report['selected_feature_set']}")
        print(f"Selected RUL candidate: {report['selected_rul_candidate']}")
        print("Development acceptance: PENDING_UNOPENED")
        print("Sealed holdout authorized: False")
    elif args.command == "evaluate-rul-v2-4-development":
        config = AppConfig.load(args.config)
        report = evaluate_rul_v2_4_development_acceptance(
            args.manifest,
            args.consumed_registry,
            args.freeze_manifest,
            args.model,
            args.frozen_v2_3_model,
            args.output_dir,
            config.sensor,
        )
        print(f"Development gates: {'PASS' if report['all_required_gates_passed'] else 'FAIL'}")
        print(f"Sealed holdout authorized: {report['sealed_holdout_authorized']}")
        print("Sealed holdout generated/evaluated: False")
    elif args.command == "generate-rul-v2-5-development":
        manifest = generate_rul_v2_5_development_corpus(
            args.data_dir,
            args.manifest,
            args.consumed_registry,
            role_seeds={
                "fit": args.fit_seeds,
                "calibration": args.calibration_seeds,
                "acceptance": args.acceptance_seeds,
            },
            lifecycles_per_batch=args.lifecycles_per_batch,
            machines=args.machines,
            cadence_seconds=args.cadence_seconds,
            line_sel=args.line_sel,
        )
        print(f"RUL v2.5 physically separated evidence: {Path(args.data_dir).resolve()}")
        for role, row in manifest["role_files"].items():
            print(f"{role}: {row['batches']} batches, {row['lifecycles']} lifecycles, {row['rows']} rows")
        print("Acceptance content remains unparsed; only its generation-time hash is recorded.")
    elif args.command == "train-rul-v2-5":
        config = AppConfig.load(args.config)
        report = train_rul_v2_5_candidate(
            args.manifest,
            args.consumed_registry,
            args.frozen_v2_4_model,
            args.model,
            args.output_dir,
            args.freeze_manifest,
            config.sensor,
        )
        print(f"Frozen RUL v2.5 artifact: {Path(args.model).resolve()}")
        print(f"Selected calibration candidate: {report['selected_calibration_candidate']}")
        print(f"Preacceptance candidate gates: {'PASS' if report['candidate_selection_passed'] else 'FAIL'}")
        print("Development acceptance: PENDING_UNOPENED")
        print("Sealed holdout authorized: False")
    elif args.command == "evaluate-rul-v2-5-development":
        config = AppConfig.load(args.config)
        report = evaluate_rul_v2_5_development_acceptance(
            args.manifest,
            args.consumed_registry,
            args.freeze_manifest,
            args.model,
            args.frozen_v2_4_model,
            args.output_dir,
            config.sensor,
        )
        print(f"Development gates: {'PASS' if report['all_required_gates_passed'] else 'FAIL'}")
        print(f"Sealed holdout authorized: {report['sealed_holdout_authorized']}")
        print("Sealed holdout generated/evaluated: False")
    elif args.command == "generate-rul-v2-6-preacceptance":
        manifest = generate_rul_v2_6_preacceptance_corpus(
            args.data_dir, args.manifest, args.consumed_registry,
            role_seeds={"fit": args.fit_seeds, "calibration": args.calibration_seeds,
                        "acceptance": args.acceptance_seeds},
            lifecycles_per_batch=args.lifecycles_per_batch, machines=args.machines,
            cadence_seconds=args.cadence_seconds, line_sel=args.line_sel,
        )
        print(f"RUL v2.6 fit/calibration evidence: {Path(args.data_dir).resolve()}")
        print(f"Acceptance: {manifest['acceptance']['status']}")
    elif args.command == "train-rul-v2-6":
        config = AppConfig.load(args.config)
        report = train_rul_v2_6_candidate(
            args.manifest, args.consumed_registry, args.frozen_v2_5_model,
            args.model, args.output_dir, args.freeze_manifest, config.sensor,
        )
        print(f"Frozen RUL v2.6 artifact: {Path(args.model).resolve()}")
        print(f"Final artifact calibration parity: {'PASS' if report['final_artifact_calibration_parity']['passed'] else 'FAIL'}")
        print(f"Preacceptance candidate gates: {'PASS' if report['preacceptance_candidate_gates_passed'] else 'FAIL'}")
        print(f"Acceptance generation authorized: {report['acceptance_generation_authorized']}")
        print("Acceptance: NOT_GENERATED")
    elif args.command == "generate-rul-v2-6-acceptance":
        manifest = generate_rul_v2_6_acceptance_after_parity(
            args.manifest, args.consumed_registry, args.freeze_manifest,
        )
        print(f"Fresh RUL v2.6 acceptance: {manifest['acceptance']['status']}")
        print("Acceptance content remains unopened.")
    elif args.command == "evaluate-rul-v2-6-development":
        config = AppConfig.load(args.config)
        report = evaluate_rul_v2_6_development_acceptance(
            args.manifest, args.consumed_registry, args.freeze_manifest,
            args.model, args.output_dir, config.sensor,
        )
        print(f"Development gates: {'PASS' if report['all_required_gates_passed'] else 'FAIL'}")
        print("Target authorization: " + json.dumps(report["target_authorization"], sort_keys=True))
        print("Sealed holdout generated/evaluated: False")
    elif args.command == "generate-rul-v2-7-preacceptance":
        manifest = generate_rul_v2_7_preacceptance_corpus(
            args.data_dir, args.manifest, args.consumed_registry,
            role_seeds={"fit": args.fit_seeds, "calibration": args.calibration_seeds,
                        "acceptance": args.acceptance_seeds},
            lifecycles_per_batch=args.lifecycles_per_batch, machines=args.machines,
            cadence_seconds=args.cadence_seconds, line_sel=args.line_sel,
        )
        print(f"RUL v2.7 fresh fit/calibration evidence: {Path(args.data_dir).resolve()}")
        print(f"Acceptance: {manifest['acceptance']['status']}")
    elif args.command == "train-rul-v2-7":
        config = AppConfig.load(args.config)
        report = train_rul_v2_7_candidate(
            args.manifest, args.consumed_registry, args.frozen_v2_6_model,
            args.model, args.output_dir, args.freeze_manifest, config.sensor,
        )
        selected = report["target_selection"]["critical"]["selected_bias_correction"]
        print(f"Frozen RUL v2.7 artifact: {Path(args.model).resolve()}")
        print(f"CRITICAL correction selected: {selected}")
        print(f"Final artifact calibration parity: {'PASS' if report['final_artifact_calibration_parity']['passed'] else 'FAIL'}")
        print(f"Preacceptance candidate gates: {'PASS' if report['preacceptance_candidate_gates_passed'] else 'FAIL'}")
        print(f"Acceptance generation authorized: {report['acceptance_generation_authorized']}")
        print("Acceptance: NOT_GENERATED")
    elif args.command == "generate-rul-v2-7-acceptance":
        manifest = generate_rul_v2_7_acceptance_after_parity(
            args.manifest, args.consumed_registry, args.freeze_manifest,
        )
        print(f"Fresh RUL v2.7 acceptance: {manifest['acceptance']['status']}")
        print("Acceptance content remains unopened.")
    elif args.command == "evaluate-rul-v2-7-development":
        config = AppConfig.load(args.config)
        report = evaluate_rul_v2_7_development_acceptance(
            args.manifest, args.consumed_registry, args.freeze_manifest,
            args.model, args.output_dir, config.sensor,
        )
        print(f"Development gates: {'PASS' if report['all_required_gates_passed'] else 'FAIL'}")
        print("Target authorization: " + json.dumps(report["target_authorization"], sort_keys=True))
        print("Sealed holdout generated/evaluated: False")
    elif args.command == "generate-rul-v2-7-sealed-holdout":
        manifest = generate_rul_v2_7_sealed_holdout(
            args.data_dir, args.manifest, args.consumed_registry,
            args.acceptance_freeze, args.model, seeds=args.seeds,
            lifecycles_per_batch=args.lifecycles_per_batch, machines=args.machines,
            cadence_seconds=args.cadence_seconds, line_sel=args.line_sel,
        )
        holdout = manifest["sealed_holdout"]
        print(f"Sealed holdout: {holdout['status']}")
        print(f"Batches: {len(holdout['seeds'])}; lifecycles: {holdout['role_file']['lifecycles']}")
        print("Content remains unopened by the evaluator.")
    elif args.command == "evaluate-rul-v2-7-sealed-holdout":
        config = AppConfig.load(args.config)
        report = evaluate_rul_v2_7_sealed_holdout(
            args.manifest, args.consumed_registry, args.model,
            args.output_dir, config.sensor,
        )
        print(f"Sealed synthetic gates: {'PASS' if report['all_required_gates_passed'] else 'FAIL'}")
        print(f"Synthetic candidate locked: {report['synthetic_candidate_locked']}")
        print("Plant production authorized: False")
    elif args.command == "train-mock":
        config = AppConfig.load(args.config)
        report = train_bootstrap_model(
            args.input, args.model, args.report, config.sensor,
            seed=args.seed, baseline_anchor_fraction=args.baseline_anchor_fraction,
        )
        test = report["test_evaluation"]
        print(f"Bootstrap regime model: {Path(args.model).resolve()}")
        print(f"Report: {Path(args.report).resolve()}")
        print(f"Test early-anchor alert proxy: {test['early_anchor_alert_rate']}")
        print(f"Test late critical coverage proxy: {test['late_critical_coverage']}")
        print(f"Test degradation/progress Spearman: {test['score_progress_spearman']}")
        print(f"Test degradation/hidden-damage Spearman: {test['score_latent_damage_spearman']}")
        rul_test = report.get("rul_model", {}).get("critical_untouched_test", {})
        if rul_test:
            print(f"Untouched-test learned RUL MAE: {rul_test.get('mae_hours')} h")
            print(f"Untouched-test learned RUL median absolute error: {rul_test.get('median_abs_error_hours')} h")
            print(f"Untouched-test learned RUL interval coverage: {rul_test.get('interval_coverage')}")
        print("WARNING: synthetic regime/RUL metrics are not real-plant production validation.")
    elif args.command == "evaluate-rul":
        config = AppConfig.load(args.config)
        report = evaluate_rul(
            args.model,
            args.input,
            args.report,
            args.predictions,
            config.sensor,
            ground_truth_path=args.ground_truth,
            criteria=RULEvaluationCriteria(
                max_critical_mae_hours=args.max_critical_mae_hours,
                max_critical_median_abs_error_hours=args.max_critical_median_abs_error_hours,
                min_critical_estimate_availability=args.min_critical_estimate_availability,
                min_rul_monotonicity=args.min_rul_monotonicity,
                min_interval_coverage=args.min_interval_coverage,
                min_macro_lifecycle_interval_coverage=args.min_macro_lifecycle_interval_coverage,
                max_within_24h_mae_hours=args.max_within_24h_mae_hours,
                max_within_12h_mae_hours=args.max_within_12h_mae_hours,
                max_within_6h_mae_hours=args.max_within_6h_mae_hours,
                max_premature_short_rul_rate=args.max_premature_short_rul_rate,
                max_sensor_fault_large_rul_jump_rate=args.max_sensor_fault_large_rul_jump_rate,
            ),
        )
        status = "PASS" if report["overall_pass"] else "FAIL"
        critical = report["pre_onset_critical_rul"]
        print(f"RUL evaluation: {status}")
        print(f"RUL method: {report.get('rul_method')}")
        print(f"Report: {Path(args.report).resolve()}")
        print(f"Predictions: {Path(args.predictions).resolve()}")
        print(f"Critical RUL MAE: {critical['mae_hours']} h")
        print(f"Critical RUL median absolute error: {critical['median_abs_error_hours']} h")
        print(f"Critical estimate availability: {critical['availability']}")
        print(f"RUL monotonicity: {critical['monotonicity']}")
        print(f"Pre-onset critical interval coverage: {critical['interval_coverage']}")
        print(f"Macro lifecycle interval coverage: {critical['macro_lifecycle_coverage']}")
        print(f"Mean/median interval width: {critical['mean_interval_width_hours']} / {critical['median_interval_width_hours']} h")
        for name, metrics in report["horizon_metrics"].items():
            print(f"{name} MAE: {metrics['mae_hours']} h; availability={metrics['availability']}")
        for name, metrics in report["anchor_metrics"].items():
            print(f"{name} anchor MAE: {metrics['mae_hours']} h; lifecycle availability={metrics['availability']}")
        sensor_fault = report["sensor_fault_stability"]
        if sensor_fault["comparable_rows"]:
            print(f"Sensor-fault large RUL jump rate: {sensor_fault['large_rul_jump_rate']}")
        failed = [name for name, passed in report["checks"].items() if not passed]
        if failed:
            print("Failed RUL checks: " + ", ".join(failed))
        print("WARNING: RUL PASS is synthetic development evidence only, not plant life validation.")
    elif args.command == "evaluate-generalization":
        config = AppConfig.load(args.config)
        report = evaluate_generalization(
            args.model,
            args.report,
            config.sensor,
            trials=args.trials,
            lifecycles=args.lifecycles,
            machines=args.machines,
            cadence_seconds=args.cadence_seconds,
            seed_start=args.seed_start,
            seeds=args.seeds,
            line_sel=args.line_sel,
            keep_generated_dir=args.keep_generated_dir,
            criteria=GeneralizationCriteria(
                max_early_anchor_alert_rate=args.max_early_alert_rate,
                min_late_critical_coverage=args.min_late_critical_coverage,
                min_score_progress_spearman=args.min_progress_spearman,
                min_score_latent_damage_spearman=args.min_hidden_damage_spearman,
                max_transition_reversal_rate=args.max_transition_reversal_rate,
            ),
        )
        status = "PASS" if report["overall_pass"] else "FAIL"
        aggregate = report["aggregate_metrics"]
        print(f"Generalization evaluation: {status}")
        print(f"Passed trials: {report['passed_trials']}/{len(report['trials'])}")
        print(f"Report: {Path(args.report).resolve()}")
        print(f"Mean early-anchor alert rate: {aggregate['early_anchor_alert_rate']['mean']}")
        print(f"Mean late critical coverage: {aggregate['late_critical_coverage']['mean']}")
        print(f"Mean degradation/progress Spearman: {aggregate['score_progress_spearman']['mean']}")
        print(f"Mean degradation/hidden-damage Spearman: {aggregate['score_latent_damage_spearman']['mean']}")
        print("WARNING: PASS is synthetic bootstrap generalization only, not plant production validation.")
    elif args.command == "generate-stress":
        result = generate_stress_suite(
            args.output_dir,
            StressSuiteConfig(
                cadence_seconds=args.cadence_seconds,
                replicates=args.replicates,
                machines=args.machines,
                line_sel=args.line_sel,
                suite_version=args.suite_version,
            ),
            overwrite=args.overwrite,
        )
        print(f"Generated frozen stress suite: {result['suite_dir']}")
        print(f"Rows: {result['rows']}; lifecycles: {result['lifecycles']}; scenarios: {result['scenarios']}")
        print(f"Sensor CSV: {result['sensor_csv']}")
        print(f"Hidden ground truth: {result['ground_truth_csv']}")
        print(f"NOTE: {args.suite_version} is retained as consumed regression evidence in this project snapshot.")
    elif args.command == "evaluate-stress":
        config = AppConfig.load(args.config)
        report = evaluate_stress_suite(
            args.model,
            args.suite_dir,
            args.report,
            config.sensor,
            criteria=StressCriteria(
                max_overall_fp_rate=args.max_overall_fp_rate,
                max_overall_fn_rate=args.max_overall_fn_rate,
                min_overall_critical_recall=args.min_overall_critical_recall,
                max_healthy_alert_rate=args.max_healthy_alert_rate,
                max_sensor_fault_alert_rate=args.max_sensor_fault_alert_rate,
                max_fault_fn_rate=args.max_fault_fn_rate,
                min_fault_critical_recall=args.min_fault_critical_recall,
                min_sudden_critical_recall=args.min_sudden_critical_recall,
            ),
        )
        status = "PASS" if report["overall_pass"] else "FAIL"
        metrics = report["overall_metrics"]
        print(f"Stress evaluation: {status}")
        print(f"Passed scenarios: {report['passed_scenarios']}/{len(report['per_scenario'])}")
        print(f"Report: {Path(args.report).resolve()}")
        print(f"Overall FP rate: {metrics['false_positive_rate']}")
        print(f"Overall FN rate: {metrics['false_negative_rate']}")
        print(f"Overall exact CRITICAL recall: {metrics['critical_exact_recall']}")
        failed = [name for name, value in report["per_scenario"].items() if not value["passed"]]
        if failed:
            print("Failed scenarios: " + ", ".join(failed))
        if report.get("cadence_warning"):
            print("WARNING: " + report["cadence_warning"])
        print(f"WARNING: {report['suite_version']} is synthetic adversarial evidence only, not plant production validation.")
    elif args.command == "monitor-postgres":
        run_monitor_postgres(args)
    elif args.command == "export-csv":
        count = export_csv(args.database, args.output)
        print(f"Exported {count} row(s) to {Path(args.output).resolve()}")
    elif args.command == "profile-local":
        run_profile_local(args)
    elif args.command == "show-latest":
        run_show_latest(args)
    elif args.command == "plant-shadow":
        run_plant_shadow_command(args)
    else:
        raise RuntimeError(args.command)
