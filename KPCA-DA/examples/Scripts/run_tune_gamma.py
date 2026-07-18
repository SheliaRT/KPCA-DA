import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from kpcada import load_expression_matrix, fit_transform, mmd_loss

# --- Data paths ---
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPTS_DIR, "..", "data", "TvsN")

SOURCE_PATH = os.path.join(DATA_DIR, "TCGA_BRCA_NvT_new.csv")
TARGET_PATH = os.path.join(DATA_DIR, "GSE45498_NvT.csv")

# --- Load data ---
print("Loading source data (TCGA)...")
X_source, genes_source = load_expression_matrix(SOURCE_PATH)
print(f"  Shape: {X_source.shape}  |  Genes: {len(genes_source)}")

print("Loading target data (GSE45498)...")
X_target, genes_target = load_expression_matrix(TARGET_PATH)
print(f"  Shape: {X_target.shape}  |  Genes: {len(genes_target)}")

# --- Fixed lambda, varying gamma ---
LAMBDA_REG = 0.01
GAMMA_VALUES = [0.5, 1, 10, 100, 500, 1000, 1895, 5000, 10000]

print("\n" + "=" * 60)
print(f"Gamma grid search  (lambda fixed at {LAMBDA_REG})")
print(f"Gamma candidates: {GAMMA_VALUES}")
print("=" * 60)

records = []
for gamma in GAMMA_VALUES:
    print(f"\n--- gamma = {gamma} ---")
    res = fit_transform(
        X_source=X_source,
        X_target=X_target,
        genes_source=genes_source,
        genes_target=genes_target,
        n_components=2,
        kernel="rbf",
        kpca_gamma=0.5,
        mmd_gamma=gamma,
        auto_gamma=False,
        lambda_reg=LAMBDA_REG,
        lr=0.001,
        epochs=500,
        verbose=True,
    )

    Z_s = res["Z_s"]
    Z_t = res["Z_t"]
    common_src = res["common_source_idx"]
    common_tgt = res["common_target_idx"]

    Zs_t = torch.tensor(Z_s[common_src], dtype=torch.float32)
    Zt_t = torch.tensor(Z_t[common_tgt], dtype=torch.float32)
    final_mmd = mmd_loss(Zs_t, Zt_t, gamma=gamma).item()
    final_loss = res["loss_history"][-1]

    print(f"  final MMD={final_mmd:.6f}  final loss={final_loss:.6f}")
    records.append({
        "gamma": gamma,
        "final_mmd": round(final_mmd, 6),
        "final_loss": round(final_loss, 6),
    })

# --- Results ---
df = pd.DataFrame(records)
print("\n" + "=" * 60)
print(f"RESULTS  (lambda = {LAMBDA_REG})")
print("=" * 60)
print(df.to_string(index=False))

best = df.loc[df["final_mmd"].idxmin()]
print(f"\nBest gamma (lowest MMD): {best['gamma']}  (MMD = {best['final_mmd']})")
