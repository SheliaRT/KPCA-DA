import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from kpcada import (
    load_expression_matrix, load_labels,
    fit_transform, tune_lambda, mmd_loss,
    compute_median_gamma, find_common_gene_indices,
)
from sklearn.decomposition import KernelPCA

# ── Load data ─────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPTS_DIR, "..", "data", "TvsN")
SRC_PATH    = os.path.join(DATA_DIR, "TCGA_BRCA_NvT_new.csv")
TGT_PATH    = os.path.join(DATA_DIR, "GSE45498_NvT.csv")

print("Loading data...")
X_source, genes_source = load_expression_matrix(SRC_PATH)
X_target, genes_target = load_expression_matrix(TGT_PATH)

# Labels (for Table 3)
y_src_raw = load_labels(SRC_PATH)
y_tgt_raw = load_labels(TGT_PATH)
mask = y_src_raw != "Unknown"
X_source_labeled = X_source[:, mask]
y_src_raw = y_src_raw[mask]
le = LabelEncoder().fit(["Normal", "Tumor"])
y_source = le.transform(y_src_raw)
y_target = le.transform(y_tgt_raw)

print(f"  Source: {X_source.shape}  Target: {X_target.shape}")

# ── Shared gamma (pre-computed once from a fixed KPCA) ────────────────────────
# Using random_state=42 on KernelPCA so the pre-computation is reproducible
common_src, common_tgt = find_common_gene_indices(genes_source, genes_target)
kpca_init_s = KernelPCA(n_components=2, kernel="rbf", gamma=0.5, random_state=42)
kpca_init_t = KernelPCA(n_components=2, kernel="rbf", gamma=0.5, random_state=42)
Zs_init = kpca_init_s.fit_transform(X_source)
Zt_init = kpca_init_t.fit_transform(X_target)
SHARED_GAMMA = compute_median_gamma(
    torch.tensor(Zs_init[common_src], dtype=torch.float32),
    torch.tensor(Zt_init[common_tgt], dtype=torch.float32),
)
print(f"\nShared MMD gamma (median heuristic, random_state=42): {SHARED_GAMMA:.4f}")

LAMBDA_FIXED = 0.01
GAMMA_VALUES  = [0.5, 1, 10, 100, 500, 1000, 1895, 5000, 10000]
LAMBDA_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
EPOCHS = 500
LR     = 0.001

# ════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Various gamma  |  MMD loss  (lambda = 0.01 fixed)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"TABLE 1: Gamma vs MMD Loss   (lambda fixed = {LAMBDA_FIXED})")
print("=" * 65)

records1 = []
for gamma in GAMMA_VALUES:
    print(f"  gamma={gamma} ...", end="", flush=True)
    res = fit_transform(
        X_source=X_source, X_target=X_target,
        genes_source=genes_source, genes_target=genes_target,
        n_components=2, kernel="rbf", kpca_gamma=0.5,
        mmd_gamma=gamma, auto_gamma=False,
        lambda_reg=LAMBDA_FIXED, lr=LR, epochs=EPOCHS, verbose=False,
    )
    Z_s, Z_t = res["Z_s"], res["Z_t"]
    src_idx, tgt_idx = res["common_source_idx"], res["common_target_idx"]
    final_mmd  = mmd_loss(
        torch.tensor(Z_s[src_idx], dtype=torch.float32),
        torch.tensor(Z_t[tgt_idx], dtype=torch.float32),
        gamma=gamma,
    ).item()
    final_loss = res["loss_history"][-1]
    print(f"  MMD={final_mmd:.6f}  Loss={final_loss:.6f}")
    records1.append({"gamma": gamma, "final_mmd": round(final_mmd, 6),
                     "final_loss": round(final_loss, 6)})

df1 = pd.DataFrame(records1)
print("\n" + df1.to_string(index=False))
best1 = df1.loc[df1["final_mmd"].idxmin()]
print(f"\n  >> Best gamma (lowest MMD): {best1['gamma']}  (MMD={best1['final_mmd']})")

# ════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Various lambda  |  MMD loss  (gamma = SHARED_GAMMA)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"TABLE 2: Lambda vs MMD Loss   (gamma fixed = {SHARED_GAMMA:.4f})")
print("=" * 65)

records2 = []
for lam in LAMBDA_VALUES:
    print(f"  lambda={lam} ...", end="", flush=True)
    res = fit_transform(
        X_source=X_source, X_target=X_target,
        genes_source=genes_source, genes_target=genes_target,
        n_components=2, kernel="rbf", kpca_gamma=0.5,
        mmd_gamma=SHARED_GAMMA, auto_gamma=False,
        lambda_reg=lam, lr=LR, epochs=EPOCHS, verbose=False,
    )
    Z_s, Z_t = res["Z_s"], res["Z_t"]
    src_idx, tgt_idx = res["common_source_idx"], res["common_target_idx"]
    final_mmd  = mmd_loss(
        torch.tensor(Z_s[src_idx], dtype=torch.float32),
        torch.tensor(Z_t[tgt_idx], dtype=torch.float32),
        gamma=SHARED_GAMMA,
    ).item()
    final_loss = res["loss_history"][-1]
    print(f"  MMD={final_mmd:.6f}  Loss={final_loss:.6f}")
    records2.append({"lambda": lam, "final_mmd": round(final_mmd, 6),
                     "final_loss": round(final_loss, 6)})

df2 = pd.DataFrame(records2)
print("\n" + df2.to_string(index=False))
best2 = df2.loc[df2["final_mmd"].idxmin()]
print(f"\n  >> Best lambda (lowest MMD): {best2['lambda']}  (MMD={best2['final_mmd']})")

# ════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Various lambda  |  Classification accuracy  (gamma = SHARED_GAMMA)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"TABLE 3: Lambda vs Classification Accuracy   (gamma fixed = {SHARED_GAMMA:.4f})")
print("=" * 65)

summary3 = tune_lambda(
    X_source=X_source_labeled, X_target=X_target,
    genes_source=genes_source, genes_target=genes_target,
    lambda_values=LAMBDA_VALUES,
    y_source=y_source, y_target=y_target,
    n_components=2, kernel="rbf", kpca_gamma=0.5,
    mmd_gamma=SHARED_GAMMA, auto_gamma=False,
    lr=LR, epochs=EPOCHS, verbose=False,
)

df3 = summary3["results_df"][["lambda", "final_mmd", "final_loss", "accuracy"]]
print("\n" + df3.to_string(index=False))
print(f"\n  >> Best lambda (highest accuracy): {summary3['best_lambda_accuracy']}")
print(f"  >> Best lambda (lowest MMD):       {summary3['best_lambda_mmd']}")
