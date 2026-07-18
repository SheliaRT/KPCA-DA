import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
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
print(f"  Source : {X_source.shape}")
print(f"  Target : {X_target.shape}")

# ── Split target → adapt (70%) / test (30%), stratified ──────────────────────
y_tgt_raw = load_labels(os.path.join(DATA_DIR, "GSE45498_NvT.csv"))
X_adapt_T, X_test_T, _, _ = train_test_split(
    X_target.T, y_tgt_raw,
    test_size=0.3, random_state=42, stratify=y_tgt_raw,
)
X_adapt = X_adapt_T.T    # genes × adapt_samples
X_test  = X_test_T.T     # genes × test_samples

print(f"\nTarget split (stratified, random_state=42):")
print(f"  Adapt : {X_adapt.shape[1]} samples  (used for adaptation)")
print(f"  Test  : {X_test.shape[1]}  samples  (held out, not used)")

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
print(f"Adaptation: Source + Adapt set   |   Metric: MMD and total loss only")
print("=" * 65)

records = []
run = 0
for gamma in GAMMA_VALUES:
    for lam in LAMBDA_VALUES:
        run += 1
        print(f"  [{run:2d}/{TOTAL}]  gamma={gamma:6}  lambda={lam} ...", end="", flush=True)

        res = fit_transform(
            X_source=X_source, X_target=X_adapt,
            genes_source=genes_source, genes_target=genes_target,
            n_components=2, kernel="rbf", kpca_gamma=0.5,
            mmd_gamma=gamma, auto_gamma=False,
            lambda_reg=lam, lr=LR, epochs=EPOCHS, verbose=False,
        )

        Z_s     = res["Z_s"]
        Z_t     = res["Z_t"]
        src_idx = res["common_source_idx"]
        tgt_idx = res["common_target_idx"]

        final_mmd = mmd_loss(
            torch.tensor(Z_s[src_idx], dtype=torch.float32),
            torch.tensor(Z_t[tgt_idx], dtype=torch.float32),
            gamma=gamma,
        ).item()
        final_loss = res["loss_history"][-1]

        print(f"  MMD={final_mmd:.6f}  Loss={final_loss:.6f}")
        records.append({
            "gamma":      gamma,
            "lambda":     lam,
            "final_mmd":  round(final_mmd,  6),
            "final_loss": round(final_loss, 6),
        })

df = pd.DataFrame(records)

# ── Grid tables ───────────────────────────────────────────────────────────────
def pivot(df, col, fmt):
    g = df.pivot(index="gamma", columns="lambda", values=col)
    g.columns = [f"lam={c}" for c in g.columns]
    g.index   = [f"g={r}"   for r in g.index]
    return g.map(lambda x: fmt.format(x))

print("\n" + "=" * 65)
print("GRID TABLE A — Final MMD Loss")
print("=" * 65)
print(pivot(df, "final_mmd", "{:.6f}").to_string())

print("\n" + "=" * 65)
print("GRID TABLE B — Final Total Loss")
print("=" * 65)
print(pivot(df, "final_loss", "{:.6f}").to_string())

# ── Best pairs ────────────────────────────────────────────────────────────────
best_mmd  = df.loc[df["final_mmd"].idxmin()]
best_loss = df.loc[df["final_loss"].idxmin()]

print("\n" + "=" * 65)
print("OPTIMAL PAIRS")
print("=" * 65)
print(f"  Best by lowest MMD   :  gamma={int(best_mmd['gamma'])}  lambda={best_mmd['lambda']}"
      f"  ->  MMD={best_mmd['final_mmd']}  Loss={best_mmd['final_loss']}")
print(f"  Best by lowest Loss  :  gamma={int(best_loss['gamma'])}  lambda={best_loss['lambda']}"
      f"  ->  MMD={best_loss['final_mmd']}  Loss={best_loss['final_loss']}")

print("\n" + "=" * 65)
print("FULL RESULTS (sorted by MMD)")
print("=" * 65)
print(df.sort_values("final_mmd").to_string(index=False))
