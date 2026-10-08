import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from kpcada import (
    load_expression_matrix, load_labels,
    fit_transform, mmd_loss, compute_median_gamma, find_common_gene_indices,
)

# ── Load data ─────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPTS_DIR, "..", "data", "TvsN")

print("Loading data...")
X_source, genes_source = load_expression_matrix(os.path.join(DATA_DIR, "TCGA_BRCA_NvT_new.csv"))
X_target, genes_target = load_expression_matrix(os.path.join(DATA_DIR, "GSE45498_NvT.csv"))

y_src_raw = load_labels(os.path.join(DATA_DIR, "TCGA_BRCA_NvT_new.csv"))
y_tgt_raw = load_labels(os.path.join(DATA_DIR, "GSE45498_NvT.csv"))

# Filter unknown source samples
mask = y_src_raw != "Unknown"
X_source = X_source[:, mask]
y_src_raw = y_src_raw[mask]

le = LabelEncoder().fit(["Normal", "Tumor"])
y_source = le.transform(y_src_raw)
y_target = le.transform(y_tgt_raw)

print(f"  Source : {X_source.shape}  ({len(y_source)} labelled samples)")
print(f"  Target : {X_target.shape}  ({len(y_target)} labelled samples)")

# ── Split target → adapt (70%) / test (30%), stratified ──────────────────────
# X_target shape: genes × samples  →  transpose for split, then transpose back
X_tgt_T = X_target.T      # samples × genes
X_adapt_T, X_test_T, y_adapt, y_test = train_test_split(
    X_tgt_T, y_target,
    test_size=0.3, random_state=42, stratify=y_target,
)
X_adapt = X_adapt_T.T     # genes × adapt_samples
X_test  = X_test_T.T      # genes × test_samples

print(f"\nTarget split (stratified, random_state=42):")
print(f"  Adapt : {X_adapt.shape[1]} samples  "
      f"(Normal={sum(y_adapt==0)}, Tumor={sum(y_adapt==1)})")
print(f"  Test  : {X_test.shape[1]}  samples  "
      f"(Normal={sum(y_test==0)},  Tumor={sum(y_test==1)})")

# ── Pre-compute shared gamma from adapt set (fixed seed) ─────────────────────
common_src, common_tgt = find_common_gene_indices(genes_source, genes_target)
kpca_s = KernelPCA(n_components=2, kernel="rbf", gamma=0.5, random_state=42)
kpca_t = KernelPCA(n_components=2, kernel="rbf", gamma=0.5, random_state=42)
Zs_init = kpca_s.fit_transform(X_source)
Zt_init = kpca_t.fit_transform(X_adapt)
SHARED_GAMMA = compute_median_gamma(
    torch.tensor(Zs_init[common_src], dtype=torch.float32),
    torch.tensor(Zt_init[common_tgt], dtype=torch.float32),
)
print(f"\nShared MMD gamma (median heuristic, random_state=42): {SHARED_GAMMA:.4f}")

# ── Grid ──────────────────────────────────────────────────────────────────────
GAMMA_VALUES  = [500, 1000, 1895, 5000, 10000]
LAMBDA_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0]
EPOCHS = 500
LR     = 0.001
TOTAL  = len(GAMMA_VALUES) * len(LAMBDA_VALUES)

print(f"\nJoint 2D grid search: {len(GAMMA_VALUES)} gamma x {len(LAMBDA_VALUES)} lambda = {TOTAL} runs")
print(f"Adaptation: Source + Adapt set   |   Evaluation: Test set (held out)")
print("=" * 75)

records = []
run = 0
for gamma in GAMMA_VALUES:
    for lam in LAMBDA_VALUES:
        run += 1
        print(f"  [{run:2d}/{TOTAL}]  gamma={gamma:6}  lambda={lam} ...", end="", flush=True)

        # ── Adapt: source + X_adapt ──────────────────────────────────────────
        res = fit_transform(
            X_source=X_source, X_target=X_adapt,
            genes_source=genes_source, genes_target=genes_target,
            n_components=2, kernel="rbf", kpca_gamma=0.5,
            mmd_gamma=gamma, auto_gamma=False,
            lambda_reg=lam, lr=LR, epochs=EPOCHS, verbose=False,
        )

        Z_s     = res["Z_s"]          # shape: n_src_genes  × 2
        Z_t     = res["Z_t"]          # shape: n_tgt_genes  × 2
        src_idx = res["common_source_idx"]
        tgt_idx = res["common_target_idx"]

        # ── MMD between source and adapt embeddings ───────────────────────────
        final_mmd = mmd_loss(
            torch.tensor(Z_s[src_idx], dtype=torch.float32),
            torch.tensor(Z_t[tgt_idx], dtype=torch.float32),
            gamma=gamma,
        ).item()
        final_loss = res["loss_history"][-1]

        # ── Project samples into gene-embedding space ─────────────────────────
        # Source samples: X_source[common genes, :].T @ Z_s[common genes]
        X_src_proj   = X_source[src_idx, :].T @ Z_s[src_idx]   # (n_src_samples, 2)

        # Test samples: use gene embedding Z_t learned from adapt set
        # X_test has the same gene ordering as X_target → tgt_idx applies
        X_test_proj  = X_test[tgt_idx, :].T   @ Z_t[tgt_idx]   # (n_test_samples, 2)

        # ── Classify: train on source, test on held-out test set ─────────────
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_src_proj, y_source)
        acc = accuracy_score(y_test, clf.predict(X_test_proj))

        print(f"  MMD={final_mmd:.5f}  Loss={final_loss:.5f}  Acc={acc:.4f}")
        records.append({
            "gamma": gamma, "lambda": lam,
            "final_mmd":  round(final_mmd,  6),
            "final_loss": round(final_loss, 6),
            "accuracy":   round(acc,        4),
        })

df = pd.DataFrame(records)

# ── Grid tables ────────────────────────────────────────────────────────────────
def pivot(df, col, fmt):
    g = df.pivot(index="gamma", columns="lambda", values=col)
    g.columns = [f"lam={c}" for c in g.columns]
    g.index   = [f"g={r}"   for r in g.index]
    return g.map(lambda x: fmt.format(x))

print("\n" + "=" * 75)
print("GRID TABLE A — MMD Loss  (source vs adapt embeddings)")
print("=" * 75)
print(pivot(df, "final_mmd", "{:.5f}").to_string())

print("\n" + "=" * 75)
print("GRID TABLE B — Classification Accuracy  (trained on source, tested on held-out test set)")
print("=" * 75)
print(pivot(df, "accuracy", "{:.4f}").to_string())

# ── Best pairs ────────────────────────────────────────────────────────────────
best_mmd = df.loc[df["final_mmd"].idxmin()]
best_acc = df.loc[df["accuracy"].idxmax()]

print("\n" + "=" * 75)
print("OPTIMAL PAIRS")
print("=" * 75)
print(f"  Best by lowest MMD  :  gamma={int(best_mmd['gamma'])}  lambda={best_mmd['lambda']}"
      f"  ->  MMD={best_mmd['final_mmd']}  Acc={best_mmd['accuracy']}")
print(f"  Best by accuracy    :  gamma={int(best_acc['gamma'])}  lambda={best_acc['lambda']}"
      f"  ->  MMD={best_acc['final_mmd']}  Acc={best_acc['accuracy']}")

print("\n" + "=" * 75)
print("FULL RESULTS (sorted by MMD)")
print("=" * 75)
print(df.sort_values("final_mmd").to_string(index=False))
