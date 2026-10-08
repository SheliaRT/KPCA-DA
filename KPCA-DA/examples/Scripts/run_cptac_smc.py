"""
Grid search for CPTAC (source) → SMC (target).
  Step 1 : Restrict to top 5,000 highly variable common genes (reduces kernel
           matrix from ~2.4 GB to ~200 MB, preventing memory crash).
  Step 2 : Auto-compute gamma once via median pairwise-distance heuristic.
  Step 3 : Lambda grid search — gamma fixed to auto value, lambda varied.
           Reports MMD loss + KNN classification accuracy on held-out test set.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from kpcada import (
    load_expression_matrix, load_labels,
    fit_transform, mmd_loss, compute_median_gamma, find_common_gene_indices,
)
from sklearn.metrics import balanced_accuracy_score

TOP_N_GENES = 5000   # top highly variable common genes — safe for KPCA memory

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPTS_DIR, "..", "Data", "cptacvssmc")
SRC_PATH    = os.path.join(DATA_DIR, "cptac_labels.csv")
TGT_PATH    = os.path.join(DATA_DIR, "smc_labels.csv")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
X_source, genes_source = load_expression_matrix(SRC_PATH)
X_target, genes_target = load_expression_matrix(TGT_PATH)
y_source_raw = load_labels(SRC_PATH)
y_target_raw = load_labels(TGT_PATH)
print(f"  Source (CPTAC) : {X_source.shape}  classes: {np.unique(y_source_raw)}")
print(f"  Target (SMC)   : {X_target.shape}  classes: {np.unique(y_target_raw)}")

# ── Clean data: replace NaN/inf first, then clip z-score outliers ─────────────
# Must replace inf BEFORE percentile computation, otherwise percentiles are inf.
# Both datasets are already z-score normalized, so |values| > 10 are outliers.
X_source = np.nan_to_num(X_source, nan=0.0, posinf=0.0, neginf=0.0)
X_target = np.nan_to_num(X_target, nan=0.0, posinf=0.0, neginf=0.0)
X_source = np.clip(X_source, -10.0, 10.0)
X_target = np.clip(X_target, -10.0, 10.0)
print("  Cleaned: NaN/inf -> 0, z-score clipped at +/-10")

# ── Common genes → top N by combined variance ─────────────────────────────────
common_src_idx, common_tgt_idx = find_common_gene_indices(genes_source, genes_target)
print(f"  All common genes : {len(common_src_idx)}")

src_var = np.var(X_source[common_src_idx, :], axis=1)
tgt_var = np.var(X_target[common_tgt_idx, :], axis=1)
combined_var = src_var + tgt_var

top_idx     = np.argsort(combined_var)[::-1][:TOP_N_GENES]
top_src_idx = common_src_idx[top_idx]
top_tgt_idx = common_tgt_idx[top_idx]
top_genes   = genes_source[top_src_idx]

print(f"  Using top {TOP_N_GENES} highly variable common genes")
print(f"  Variance range (kept): {combined_var[top_idx[-1]]:.4f} – {combined_var[top_idx[0]]:.4f}")

# Subset both datasets to those genes only
X_src_top = X_source[top_src_idx, :]
X_tgt_top = X_target[top_tgt_idx, :]

# ── Split target → adapt (70%) / test (30%), stratified ──────────────────────
X_adp_T, X_tst_T, y_adp_raw, y_tst_raw = train_test_split(
    X_tgt_top.T, y_target_raw,
    test_size=0.3, random_state=42, stratify=y_target_raw,
)
X_adapt = X_adp_T.T   # (5000, n_adapt)
X_test  = X_tst_T.T   # (5000, n_test)
print(f"\nTarget split (stratified 70/30):")
print(f"  Adapt : {X_adapt.shape[1]} samples")
print(f"  Test  : {X_test.shape[1]}  samples  (held out)")

# ── Normalise (each domain independently) ────────────────────────────────────
mms_src = MinMaxScaler()
X_src_norm = mms_src.fit_transform(X_src_top.T).T   # (5000, n_src)

mms_tgt = MinMaxScaler()
X_adp_norm = mms_tgt.fit_transform(X_adapt.T).T     # fit on adapt
X_tst_norm = mms_tgt.transform(X_test.T).T          # same scaler for test

# ── Since we already restricted to common genes, create dummy same-gene arrays
# fit_transform uses find_common_gene_indices internally; pass the same gene
# list for both so it returns all indices (no further filtering needed).
genes_common = top_genes   # same names on both sides after subsetting

# ── Auto-compute gamma from initial KPCA embeddings ──────────────────────────
print("\nComputing auto gamma (median heuristic)...")
kpca_init = KernelPCA(n_components=2, kernel="rbf", gamma=None, random_state=42)
Zs_init = kpca_init.fit_transform(X_src_norm)   # (5000, 2)
Zt_init = kpca_init.fit_transform(X_adp_norm)   # (5000, 2)
# All genes are common now — use all
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
print("=" * 65)
print(f"{'Lambda':>10}  {'Auto Gamma':>12}  {'Final MMD':>11}  {'Test Acc':>10}  {'Balanced Acc':>12}")
print("-" * 65)

records = []
for lam in LAMBDA_VALUES:
    res = fit_transform(
        X_source=X_src_norm, X_target=X_adp_norm,
        genes_source=genes_common, genes_target=genes_common,
        n_components=2, kernel="rbf", kpca_gamma=None,
        mmd_gamma=AUTO_GAMMA, auto_gamma=False,
        lambda_reg=lam, lr=LR, epochs=EPOCHS, verbose=False,
    )

    Z_s   = res["Z_s"]   # (5000, 2)
    Z_t   = res["Z_t"]   # (5000, 2)
    s_idx = res["common_source_idx"]
    t_idx = res["common_target_idx"]

    # Final MMD
    final_mmd = mmd_loss(
        torch.tensor(Z_s[s_idx], dtype=torch.float32),
        torch.tensor(Z_t[t_idx], dtype=torch.float32),
        gamma=AUTO_GAMMA,
    ).item()

    # KNN: project samples via gene embedding
    # X (genes × samples).T @ Z (genes × 2) → (samples × 2)
    X_src_proj = X_src_norm.T @ Z_s           # (n_src, 2)
    X_tst_proj = X_tst_norm.T @ Z_t           # (n_test, 2) — test uses adapted Z_t

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_src_proj, y_source_raw)
    preds        = clf.predict(X_tst_proj)
    test_acc     = accuracy_score(y_tst_raw, preds)
    bal_acc      = balanced_accuracy_score(y_tst_raw, preds)

    print(f"{lam:>10}  {AUTO_GAMMA:>12.4f}  {final_mmd:>11.6f}  {test_acc:>10.4f}  {bal_acc:>12.4f}")
    records.append({
        "lambda":     lam,
        "auto_gamma": round(AUTO_GAMMA, 4),
        "final_mmd":  round(final_mmd,  6),
        "test_acc":   round(test_acc,   4),
        "bal_acc":    round(bal_acc,    4),
    })

df = pd.DataFrame(records)

# ── Summary ───────────────────────────────────────────────────────────────────
best_mmd = df.loc[df["final_mmd"].idxmin()]
best_acc = df.loc[df["test_acc"].idxmax()]
best_bal = df.loc[df["bal_acc"].idxmax()]

print("\n" + "=" * 65)
print("OPTIMAL LAMBDA")
print("=" * 65)
print(f"  Best by lowest MMD      : lambda={best_mmd['lambda']}"
      f"  ->  MMD={best_mmd['final_mmd']}  Acc={best_mmd['test_acc']}  BalAcc={best_mmd['bal_acc']}")
print(f"  Best by highest Acc     : lambda={best_acc['lambda']}"
      f"  ->  MMD={best_acc['final_mmd']}  Acc={best_acc['test_acc']}  BalAcc={best_acc['bal_acc']}")
print(f"  Best by highest BalAcc  : lambda={best_bal['lambda']}"
      f"  ->  MMD={best_bal['final_mmd']}  Acc={best_bal['test_acc']}  BalAcc={best_bal['bal_acc']}")
print(f"\n  Auto gamma (fixed for all runs) : {AUTO_GAMMA:.4f}")
print(f"  Genes used : top {TOP_N_GENES} highly variable common genes")
