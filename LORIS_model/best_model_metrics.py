import os
import glob
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
)

# 0) Auto-detect your base model directory
for candidate in ("ag_models", "AutogluonModels"):
    if os.path.isdir(candidate):
        BASE_MODEL_DIR = candidate
        break
else:
    raise FileNotFoundError("Neither 'ag_models/' nor 'AutogluonModels/' exists in the cwd")

# 1) Locate the latest timestamped folder that actually holds predictor.pkl
model_dirs = sorted(
    d for d in glob.glob(os.path.join(BASE_MODEL_DIR, "*"))
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "predictor.pkl"))
)
if not model_dirs:
    raise FileNotFoundError(f"No predictor.pkl found under {BASE_MODEL_DIR}/")
model_path = model_dirs[-1]
print(f"Loading predictor from: {model_path!r}")

# 2) Load the predictor
predictor = TabularPredictor.load(model_path)

# 3) Files to evaluate
data_files = {
    "train": "Chowell_train_Response.tsv",
    "test":  "Chowell_test_Response.tsv"
}

# 4) Loop over splits and compute metrics
rows = []
MODEL_NAME = "NeuralNetTorch_r87_BAG_L1"
# Determine which column holds the "positive" class probability
pos_class = predictor.class_labels[1]

for split, filepath in data_files.items():
    # a) Read TSV
    df = pd.read_csv(filepath, sep="\t")
    label = predictor.label
    if label not in df.columns:
        raise KeyError(f"Label column '{label}' not found in {filepath!r}")
    X = df.drop(columns=[label])
    y_true = df[label]

    # b) Get predictions & probabilities
    y_pred = predictor.predict(X, model=MODEL_NAME)
    proba_df = predictor.predict_proba(X, model=MODEL_NAME)
    y_proba = proba_df[pos_class]

    # c) Compute individual metrics
    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    prec    = precision_score(y_true, y_pred)
    rec     = recall_score(y_true, y_pred)   # Sensitivity
    f1      = f1_score(y_true, y_pred)
    pr_auc  = average_precision_score(y_true, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)

    # d) Geometric means
    gm4        = (roc_auc * pr_auc * acc * f1) ** (1/4)
    gm3_no_acc = (roc_auc * pr_auc * f1) ** (1/3)
    gm2_auc_f1 = (roc_auc * f1) ** (1/2)

    rows.append({
        "Split": split,
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "ROC_AUC": roc_auc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "PR_AUC": pr_auc,
        "Specificity": spec,
        "Kappa": kappa,
        "MCC": mcc,
        "Geometric mean of four metrics (AUC, AUPRC, accuracy and F1-score)": gm4,
        "Geometric mean of three metrics (AUC, AUPRC, and F1-score) * NO ACCURACY": gm3_no_acc,
        "Geometric mean of two metrics (AUC and F1-Score)": gm2_auc_f1
    })

# 5) Print a table
metrics_df = pd.DataFrame(rows).set_index("Split")
print(metrics_df.to_markdown(floatfmt=".4f"))

