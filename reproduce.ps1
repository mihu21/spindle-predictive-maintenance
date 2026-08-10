$ErrorActionPreference = "Stop"
$python = ".\.venv-win\Scripts\python.exe"

& $python -c "import sys; assert sys.version_info[:3] == (3, 14, 6), 'Python 3.14.6 is required by requirements-lock.txt'"
if ($LASTEXITCODE -ne 0) { throw "Python version check failed" }
& $python -m pip install --requirement requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw "Pinned dependency installation failed" }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Installed dependency compatibility check failed" }

& $python -m unittest discover -s tests -v 2> output\test_output.txt
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed with exit code $LASTEXITCODE" }
& $python -m compileall -q src tests main.py evaluate.py validate_compare.py build_portable_zip.py augment_environment_metadata.py verify_final_artifacts.py *> output\compile_output.txt
if ($LASTEXITCODE -ne 0) { throw "Static compilation failed with exit code $LASTEXITCODE" }
& $python main.py profile-data --input data\spindle_predictive_maintenance_10000_unlabeled.csv --output output\realistic\actual_data_profile.json --models-root models\realistic
& $python main.py generate-realistic-mock --profile output\realistic\actual_data_profile.json --lifecycles 30 --output data\realistic_spindle_mock.csv --metadata-output data\realistic_spindle_mock_metadata.json --seed 42
& $python main.py train-model --input data\realistic_spindle_mock.csv --data-domain realistic_synthetic --models-root models\realistic --metrics output\realistic\model_metrics.json --database output\realistic\training.db
& $python main.py evaluate-model --models-root models\realistic --output output\realistic\model_evaluation.json --database output\realistic\monitor.db
& $python main.py evaluate-guardrails --input data\realistic_spindle_mock.csv --models-root models\realistic --output output\realistic\guardrail_evaluation.json
& $python main.py replay --input data\realistic_spindle_mock.csv --data-domain realistic_synthetic --models-root models\realistic --csv output\realistic\replay.csv --lifecycles-csv output\realistic\lifecycles.csv --invalid-csv output\realistic\invalid_rows.csv --database output\realistic\monitor.db --progress-every 50000
& $python main.py evaluate-model --models-root models\realistic --promote --output output\realistic\promotion_refusal.json --database output\realistic\monitor.db

& $python main.py train-model --input data\spindle_predictive_maintenance_mock.csv --data-domain accelerated_mock --models-root models\mock --metrics output\mock\model_metrics.json --database output\mock\training.db
& $python main.py evaluate-model --models-root models\mock --output output\mock\model_evaluation.json --database output\mock\monitor.db
& $python main.py replay --input data\spindle_predictive_maintenance_mock.csv --data-domain accelerated_mock --models-root models\mock --csv output\mock\replay.csv --lifecycles-csv output\mock\lifecycles.csv --invalid-csv output\mock\invalid_rows.csv --database output\mock\monitor.db --progress-every 20000
& $python main.py evaluate-model --models-root models\mock --promote --output output\mock\promotion_refusal.json --database output\mock\monitor.db

& .\build_packages.ps1
