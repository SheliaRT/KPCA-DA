import sys
import os

# Add the Scripts folder to path so kpcada can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.preprocessing import LabelEncoder
from kpcada import load_expression_matrix, load_labels, tune_lambda, KPCA_DA

# --- Data paths ---
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPTS_DIR, "..", "data", "TvsN")

SOURCE_PATH = os.path.join(DATA_DIR, "TCGA_BRCA_NvT_new.csv")
TARGET_PATH = os.path.join(DATA_DIR, "GSE45498_NvT.csv")

# --- Load data and labels ---
print("Loading source data (TCGA)...")
X_source, genes_source = load_expression_matrix(SOURCE_PATH)
y_source_raw = load_labels(SOURCE_PATH)
print(f"  Shape: {X_source.shape}  |  Genes: {len(genes_source)}")
print(f"  Labels: {np.unique(y_source_raw, return_counts=True)}")

print("Loading target data (GSE45498)...")
X_target, genes_target = load_expression_matrix(TARGET_PATH)
y_target_raw = load_labels(TARGET_PATH)
print(f"  Shape: {X_target.shape}  |  Genes: {len(genes_target)}")
print(f"  Labels: {np.unique(y_target_raw, return_counts=True)}")

# Filter out "Unknown" source samples before encoding
known_mask = y_source_raw != "Unknown"
X_source = X_source[:, known_mask]
y_source_raw = y_source_raw[known_mask]
print(f"  Filtered source to {known_mask.sum()} known samples (removed {(~known_mask).sum()} Unknown)")

# Encode labels (Normal=0, Tumor=1)
le = LabelEncoder()
le.fit(["Normal", "Tumor"])
y_source = le.transform(y_source_raw)
y_target = le.transform(y_target_raw)
print(f"\nLabel mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- Lambda search ---
LAMBDA_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

print("\n" + "=" * 60)
print("Running lambda grid search...")
print(f"Lambda candidates: {LAMBDA_VALUES}")
print("=" * 60)

summary = tune_lambda(
    X_source=X_source,
    X_target=X_target,
    genes_source=genes_source,
    genes_target=genes_target,
    lambda_values=LAMBDA_VALUES,
    y_source=y_source,
    y_target=y_target,
    n_components=2,
    kernel="rbf",
    kpca_gamma=0.5,
    mmd_gamma=0.5,
    lr=0.001,
    epochs=500,
    verbose=True,
)

# --- Results ---
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(summary["results_df"].to_string(index=False))
print(f"\nBest lambda (lowest MMD):      {summary['best_lambda_mmd']}")
print(f"Best lambda (best accuracy):   {summary['best_lambda_accuracy']}")

# --- Fit final model with best lambda ---
best_lambda = summary["best_lambda_accuracy"]
print(f"\nFitting final model with lambda = {best_lambda} ...")

model = KPCA_DA(
    n_components=2,
    kernel="rbf",
    kpca_gamma=0.5,
    mmd_gamma=0.5,
    lambda_reg=best_lambda,
    lr=0.001,
    epochs=500,
    verbose=False,
)
Z_source, Z_target = model.fit_transform(
    X_source=X_source,
    X_target=X_target,
    genes_source=genes_source,
    genes_target=genes_target,
)

print(f"Done.")
print(f"  Z_source shape : {Z_source.shape}")
print(f"  Z_target shape : {Z_target.shape}")
print(f"  Common genes   : {model.common_source_idx_.shape[0]}")
print(f"  Final loss     : {model.loss_history_[-1]:.6f}")
