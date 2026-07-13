`make evaluate` -> `ml-lab evaluate-smoke`. Must run after `train-smoke`, and says so if you do not: a missing `reports/smoke/latest/train_run_id.txt` raises `FileNotFoundError` with "run train first".

It reads that run id back off disk, evaluates the **saved** model against the **saved** test split (deliberately loading both from disk, so a broken save/load round-trip cannot pass silently), and writes `metrics.json` plus a per-row `predictions.csv` (`y_true`, `y_pred`, `y_score`).

Then the key move: it **reopens the same MLflow run by id** and appends accuracy, precision, recall, and F1 to it. Training and evaluation are not two runs — they are one run, written in two passes, which is what makes a single MLflow row tell the whole story of a model.

Echoes the full metrics dict.
