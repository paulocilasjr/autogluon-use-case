import pandas as pd
from pycaret.classification import (
    setup,
    compare_models,
    tune_model,
    finalize_model,
    predict_model,
    save_model,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

# ──────── 1) Load data ─────────
train_df = pd.read_csv("Chowell_train_Response.tsv", sep="\t")
test_df  = pd.read_csv("Chowell_test_Response.tsv",  sep="\t")

# If your target column has a different name, change this:
TARGET = "Response"

# ──────── 2) PyCaret setup ───────
exp = setup(
    data=train_df,
    target=TARGET,
    session_id=42,
)

# ──────── 3) Baseline comparison ──
best = compare_models()

# ──────── 4) Hyperparameter tuning ─
tuned = tune_model(best)

# ──────── 5) Finalize & save ──────
final_model = finalize_model(tuned)
save_model(final_model, "pycaret_final_model")

# ──────── 6) Predict on test ──────
preds = predict_model(final_model, data=test_df)

# ──────── 7) Compute & print metrics ─
y_true  = test_df[TARGET]
y_pred  = preds["Label"]
y_proba = preds["Score"]   # positive‐class probability

acc      = accuracy_score(y_true, y_pred)
bal_acc  = balanced_accuracy_score(y_true, y_pred)
roc_auc  = roc_auc_score(y_true, y_proba)
prec     = precision_score(y_true, y_pred)
rec      = recall_score(y_true, y_pred)
f1       = f1_score(y_true, y_pred)
pr_auc   = average_precision_score(y_true, y_proba)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
spec     = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

print("\nTest-set performance on final tuned model:")
print(f"  Accuracy:           {acc:.4f}")
print(f"  Balanced Accuracy:  {bal_acc:.4f}")
print(f"  ROC AUC:            {roc_auc:.4f}")
print(f"  Precision:          {prec:.4f}")
print(f"  Recall (Sensitivity): {rec:.4f}")
print(f"  F1 Score:           {f1:.4f}")
print(f"  PR AUC:             {pr_auc:.4f}")
print(f"  Specificity:        {spec:.4f}")

