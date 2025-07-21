#!/usr/bin/env python3
"""
autogluon_multimodal.py

Run a Multimodal AutoGluon experiment on TCGA data.
Targets `er_status_by_ihc`, uses `filename` for images,
and splits by `sample` to avoid leakage:
  • 70% train
  • 10% validation
  • 20% test
"""

import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from autogluon.multimodal import MultiModalPredictor

# ─── CONFIG ──────────────────────────────────────────────────────────────────────
DATA_CSV    = "TCGA_Samples_table.csv"         # your metadata table
OUTPUT_DIR  = "ag_multimodal_output"          # where logs & models go
LABEL_COL   = "er_status_by_ihc"              # target column
GROUP_COL   = "sample"                        # grouping key to avoid leakage
IMAGE_COL   = "filename"                      # column with image file paths
RANDOM_SEED = 42

# ─── LOAD DATA ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV)
# (Optionally, if your filenames are relative to an images/ folder:)
# df[IMAGE_COL] = df[IMAGE_COL].apply(lambda fn: os.path.join("images", fn))

# ─── SPLIT: Train+Val vs Test ────────────────────────────────────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
train_val_idx, test_idx = next(gss.split(df, groups=df[GROUP_COL]))
df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
df_test      = df.iloc[test_idx].reset_index(drop=True)

# ─── SPLIT: Train vs Val ─────────────────────────────────────────────────────────
# Want 10% total val = 10/80 = 0.125 of train_val
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=RANDOM_SEED)
train_idx, val_idx = next(gss2.split(df_train_val, groups=df_train_val[GROUP_COL]))
df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
df_val   = df_train_val.iloc[val_idx].reset_index(drop=True)

print(f"▶︎ Splits:\n  Train: {len(df_train)} rows\n  Val:   {len(df_val)} rows\n  Test:  {len(df_test)} rows")

# ─── TRAIN ───────────────────────────────────────────────────────────────────────
predictor = MultiModalPredictor(
    label=LABEL_COL,
    path=OUTPUT_DIR,
    eval_metric="accuracy",  # you can choose other metrics like 'roc_auc'
)

predictor.fit(
    train_data=df_train,
    tuning_data=df_val,
    time_limit=None,         # e.g. seconds budget, or leave None for full run
    hyperparameters={        # adjust to suit your resources!
        "model.image_finetune.method": "full",   # fine‐tune entire vision model
        "env.num_workers": 8,                     # data loaders
        "optimization.max_epochs": 10,            # total training epochs
    },
    seed=RANDOM_SEED,
    image_column=IMAGE_COL,
)

# ─── EVALUATION ─────────────────────────────────────────────────────────────────
print("\n▶︎ Validation metrics:")
val_metrics = predictor.evaluate(df_val, auxiliary_metrics=True)
print(val_metrics)

print("\n▶︎ Test  metrics:")
test_metrics = predictor.evaluate(df_test, auxiliary_metrics=True)
print(test_metrics)

# ─── DETAILED LEADERBOARD ON TEST ────────────────────────────────────────────────
print("\n▶︎ Test  leaderboard (all models & key metrics):")
leaderboard = predictor.leaderboard(
    df_test,
    extra_metrics=["accuracy", "f1", "roc_auc", "precision", "recall"]
)
print(leaderboard)
