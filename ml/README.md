# Dataset preparation, training and evaluation

These scripts are optional. They do not execute when the desktop application starts.

1. `motionlab_kaggle_all_in_one.py` prepares the MM-Fit pose caches and labelled window index, with checkpoint recovery and the recorded background-review decisions. Supply the original RGB videos and labels as Kaggle inputs; review its input preflight before a full run. It uses its embedded tool payload and a CPU pose environment.
2. `train_motionlite.py` trains plain TCN and MotionLite from prepared caches on a GPU environment. `MOTIONLITE_DATA` can select the prepared dataset folder. Its source records the fixed class order, partition roles, feature selection, normalization and training settings.
3. `evaluate_motionlite.py` evaluates the frozen models. `MOTIONLITE_DATA` selects the prepared data and `MOTIONLITE_MODELS` selects a directory containing the original training result structure. Restore `evidence/checkpoints/` with the saved best checkpoints/protocols as the model root when supplying these artifacts to that environment.

The included result files are selected original artifacts from the completed runs. `evidence/results/origin.json` records the original archive hashes and selected members. Best checkpoints are retained under `evidence/checkpoints/`; last checkpoints, optimizer state and full prediction exports are excluded. The large original videos and extracted feature caches are not redistributed here.

Recorded training runtime: Python 3.12, NumPy 1.26.4 and PyTorch 2.11.0+cu128 on a Tesla T4. The app requirements are inference dependencies, not a complete training environment. The preparation script manages its separate MediaPipe runtime; PyTorch/ONNX export require a suitable separate environment.

Optional export: from an environment with PyTorch, ONNX, NumPy and ONNX Runtime, run `python scripts/export_exercise_model.py`. This reads the saved plain TCN checkpoint, verifies its hash, exports the graph and runs its existing parity checks. The exported classifier is already included, so running this step is unnecessary for normal app use.

The held-out results apply to the recorded fixed models and data selection. Reusing current test failures to tune a new model requires a fresh untouched final evaluation set.
