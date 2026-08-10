V6.0.1 evaluator semantics patch

Fixes event evaluation so WARNING and CRITICAL are counted once per lifecycle: the first confirmed target onset.
Later status fallbacks/re-entries inside the same lifecycle are not treated as independent failures and do not create new positive prediction windows.

Replace evaluate_prognostics_v6.py in the v6 project root.
The added test file is optional but recommended.

Then rerun:
  .\run_prognostics_v6.ps1

or only re-evaluate an existing replay:
  .\.venv-win\Scripts\python.exe .\evaluate_prognostics_v6.py --replay .\output\prognostics_v6\replay.csv --output .\output\prognostics_v6\evaluation.json
