# `thrember_lite` — How to run it

Five commands, in order. Replace paths/filenames as needed.

## 0. Setup

Activate the repo's venv. Then either install editable:
```bash
pip install -e .
```
or prefix every command with `PYTHONPATH=src` (no install required).

Examples below assume `PYTHONPATH=src` and Windows venv path `.venv/Scripts/python.exe`.

---

## 1. Put your JSONL files in a folder

```
my_dataset/
├── *_train_*.jsonl
├── *_test_*.jsonl
└── *_challenge_*.jsonl
```

The filename just needs to contain `train`, `test`, or `challenge` as a substring. Multiple JSONL files per subset are fine — they're concatenated.

## 2. Vectorize JSONL → `.dat` (thrember step)

```python
# scripts/vectorize.py
from thrember import create_vectorized_features
create_vectorized_features("my_dataset/")
```

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/vectorize.py
```

Produces `X_<subset>.dat` and `y_<subset>.dat` in `my_dataset/`. Done once per dataset.

## 3. Build a spec from your drop list

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m thrember_lite.cli build-spec \
    --drop dropped_features.json \
    --out runs/exp01/spec.json
```

## 4. Train

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m thrember_lite.cli train \
    my_dataset/ \
    runs/exp01/spec.json \
    runs/exp01/ \
    --config lgbm_config.json \
    --seed 42
```

Writes `model.txt` and `spec.json` into `runs/exp01/`. Use the same `--seed` across configs when comparing feature sets.

`lgbm_config.json` is optional. Minimal example:
```json
{"objective": "binary", "num_iterations": 500, "verbose": -1}
```

## 5. Predict on a PE file

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m thrember_lite.cli predict \
    runs/exp01/ \
    suspicious.exe
```

Prints `<path>\t<score>` to stdout.

---

## Notes


- **For ablation studies**, hold `--seed` constant across configs. AUC differences from feature drops are typically the same magnitude as RNG noise.
