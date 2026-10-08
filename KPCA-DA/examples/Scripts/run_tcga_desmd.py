"""
Grid search for TCGA-BRCA (source) -> DESMD (target).
  Labels : survival outcome (long / short) — binary classification.
  Step 1 : Restrict to top 5,000 highly variable common genes.
  Step 2 : Auto-compute gamma via median pairwise-distance heuristic.
  Step 3 : Lambda grid search — MMD loss + KNN accuracy on held-out test set.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler

from kpcada import (
    load_expression_matrix, load_labels,
    fit_transform, mmd_loss, compute_median_gamma, find_common_gene_indices,
)

TOP_N_GENES = 5000

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPTS_DIR, "..", "Data", "TCGAvsDesmd")
SRC_PATH    = os.path.join(DATA_DIR, "modified_TCGA_BRCA_log2.csv")
TGT_PATH    = os.path.join(DATA_DIR, "filtered_desmd_average.csv")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
X_source, genes_source = load_expression_matrix(SRC_PATH)
X_target, genes_target = load_expression_matrix(TGT_PATH)
y_source_raw = load_labels(SRC_PATH)
y_target_raw = load_labels(TGT_PATH)
print(f"  Source (TCGA) : {X_source.shape}  classes: {np.unique(y_source_raw)}")
print(f"  Target (DESMD): {X_target.shape}  classes: {np.unique(y_target_raw)}")

# ── Clean: replace NaN/inf, then clip extremes ────────────────────────────────
# Source has -33.22 floor values (log2 of zero); clipping at +/-10 handles them.
X_source = np.nan_to_num(X_source, nan=0.0, posinf=0.0, neginf=0.0)
X_target = np.nan_to_num(X_target, nan=0.0, posinf=0.0, neginf=0.0)
X_source = np.clip(X_source, -10.0, 10.0)
X_target = np.clip(X_target, -10.0, 10.0)
print("  Cleaned: NaN/inf -> 0, clipped at +/-10")

# ── Common genes -> top N by combined variance ─────────────────────────────────
common_src_idx, common_tgt_idx = find_common_gene_indices(genes_source, genes_target)
print(f"  All common genes : {len(common_src_idx)}")

src_var      = np.var(X_source[common_src_idx, :], axis=1)
tgt_var      = np.var(X_target[common_tgt_idx, :], axis=1)
combined_var = src_var + tgt_var

top_idx     = np.argsort(combined_var)[::-1][:TOP_N_GENES]
top_src_idx = common_src_idx[top_idx]
top_tgt_idx = common_tgt_idx[top_idx]
top_genes   = genes_source[top_src_idx]

print(f"  Using top {TOP_N_GENES} highly variable common genes")
print(f"  Variance range (kept): {combined_var[top_idx[-1]]:.4f} - {combined_var[top_idx[0]]:.4f}")

X_src_top = X_source[top_src_idx, :]   # (5000, n_src)
X_tgt_top = X_target[top_tgt_idx, :]   # (5000, n_tgt)

# ── Split target -> adapt (70%) / test (30%), stratified ──────────────────────
X_adp_T, X_tst_T, y_adp_raw, y_tst_raw = train_test_split(
    X_tgt_top.T, y_target_raw,
    test_size=0.3, random_state=42, stratify=y_target_raw,
)
X_adapt = X_adp_T.T
X_test  = X_tst_T.T
print(f"\nTarget split (stratified 70/30):")
print(f"  Adapt : {X_adapt.shape[1]} samples")
print(f"  Test  : {X_test.shape[1]}  samples  (held out)")
print(f"  Test class distribution: { {c: int((y_tst_raw==c).sum()) for c in np.unique(y_tst_raw)} }")

# ── Normalise (MinMaxScaler per domain, fit on adapt only for target) ─────────
mms_src = MinMaxScaler()
X_src_norm = mms_src.fit_transform(X_src_top.T).T

mms_tgt = MinMaxScaler()
X_adp_norm = mms_tgt.fit_transform(X_adapt.T).T
X_tst_norm = mms_tgt.transform(X_test.T).T

# ── Auto-compute gamma from initial KPCA embeddings ──────────────────────────
print("\nComputing auto gamma (median heuristic)...")
kpca_init = KernelPCA(n_components=2, kernel="rbf", gamma=None, random_state=42)
Zs_init   = kpca_init.fit_transform(X_src_norm)   # (5000, 2)
Zt_init   = kpca_init.fit_transform(X_adp_norm)   # (5000, 2)
AUTO_GAMMA = compute_median_gamma(
    torch.tensor(Zs_init, dtype=torch.float32),
    torch.tensor(Zt_init, dtype=torch.float32),
)
print(f"  Auto gamma : {AUTO_GAMMA:.4f}")

# ── Lambda grid search ────────────────────────────────────────────────────────
LAMBDA_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0]
EPOCHS = 500
LR     = 0.001

print(f"\nLambda grid search  (gamma fixed = {AUTO_GAMMA:.4f},  top {TOP_N_GENES} genes)")
print(f"Adaptation : Source + Adapt   |   Evaluation : KNN on held-out Test set")
print("=" * 70)
print(f"{'Lambda':>10}  {'Auto Gamma':>12}  {'Final MMD':>11}  {'Test Acc':>10}  {'Bal Acc':>9}")
print("-" * 60)

records = []
for lam in LAMBDA_VALUES:
    res = fit_transform(
        X_source=X_src_norm, X_target=X_adp_norm,
        genes_source=top_genes, genes_target=top_genes,
        n_components=2, kernel="rbf", kpca_gamma=None,
        mmd_gamma=AUTO_GAMMA, auto_gamma=False,
        lambda_reg=lam, lr=LR, epochs=EPOCHS, verbose=False,
    )

    Z_s   = res["Z_s"]
    Z_t   = res["Z_t"]
    s_idx = res["common_source_idx"]
    t_idx = res["common_target_idx"]

    final_mmd = mmd_loss(
        torch.tensor(Z_s[s_idx], dtype=torch.float32),
        torch.tensor(Z_t[t_idx], dtype=torch.float32),
        gamma=AUTO_GAMMA,
    ).item()

    # Project samples: X (genes x samples).T @ Z (genes x 2) -> (samples x 2)
    X_src_proj = X_src_norm.T @ Z_s
    X_tst_proj = X_tst_norm.T @ Z_t

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_src_proj, y_source_raw)
    preds   = clf.predict(X_tst_proj)
    acc     = accuracy_score(y_tst_raw, preds)
    bal_acc = balanced_accuracy_score(y_tst_raw, preds)

    print(f"{lam:>10}  {AUTO_GAMMA:>12.4f}  {final_mmd:>11.6f}  {acc:>10.4f}  {bal_acc:>9.4f}")
    records.append({
        "lambda":     lam,
        "auto_gamma": round(AUTO_GAMMA, 4),
        "final_mmd":  round(final_mmd, 6),
        "test_acc":   round(acc,        4),
        "bal_acc":    round(bal_acc,    4),
    })

df = pd.DataFrame(records)

best_mmd = df.loc[df["final_mmd"].idxmin()]
best_acc = df.loc[df["test_acc"].idxmax()]
best_bal = df.loc[df["bal_acc"].idxmax()]

print("\n" + "=" * 70)
print("OPTIMAL LAMBDA")
print("=" * 70)
print(f"  Best by lowest MMD     : lambda={best_mmd['lambda']}"
      f"  ->  MMD={best_mmd['final_mmd']}  Acc={best_mmd['test_acc']}  BalAcc={best_mmd['bal_acc']}")
print(f"  Best by highest Acc    : lambda={best_acc['lambda']}"
      f"  ->  MMD={best_acc['final_mmd']}  Acc={best_acc['test_acc']}  BalAcc={best_acc['bal_acc']}")
print(f"  Best by highest BalAcc : lambda={best_bal['lambda']}"
      f"  ->  MMD={best_bal['final_mmd']}  Acc={best_bal['test_acc']}  BalAcc={best_bal['bal_acc']}")
print(f"\n  Auto gamma (fixed) : {AUTO_GAMMA:.4f}")
print(f"  Genes used         : top {TOP_N_GENES} highly variable common genes")
