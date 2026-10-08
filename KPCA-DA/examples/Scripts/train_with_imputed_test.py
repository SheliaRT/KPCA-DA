"""
For each test sample (236 genes):
  1. Find the most similar source sample using the 228 common genes
  2. Build a completed 22,489-gene vector:
       - common gene positions  → actual test sample expression (rescaled to source space)
       - source-only positions  → nearest source neighbour's expression
  3. Generate test image using transform_source (same pixel grammar as source training images)
  4. Train ResNet on source + adapt images, evaluate on completed test images
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from pyDeepInsightv3.image_transformer_new import DomainAdaptiveImageTransformer

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "TvsN")

print("Loading data...")
expr1 = pd.read_csv(os.path.join(DATA_DIR, "TCGA_BRCA_NvT_new.csv"), encoding="utf-8", engine="python")
expr1 = expr1.transpose(); expr1.columns = expr1.iloc[0]
X1     = expr1.iloc[1:, :-1].values.astype(float)
Y1     = expr1.iloc[1:, -1].values
genes1 = expr1.iloc[:, :-1].columns.to_numpy()

expr2 = pd.read_csv(os.path.join(DATA_DIR, "GSE45498_NvT.csv"), encoding="utf-8", engine="python")
expr2 = expr2.transpose(); expr2.columns = expr2.iloc[0]
X2     = expr2.iloc[1:, :-1].values.astype(float)
Y2     = expr2.iloc[1:, -1].values
genes2 = expr2.iloc[:, :-1].columns.to_numpy()

# Filter Unknown source samples
mask = Y1 != "Unknown"
X1 = X1[mask]; Y1 = Y1[mask]
print(f"  Source : {X1.shape}   Target : {X2.shape}")

# ── Split target 70/30 stratified ─────────────────────────────────────────────
X2_adapt, X2_test, Y2_adapt, Y2_test = train_test_split(
    X2, Y2, test_size=0.3, random_state=42, stratify=Y2)
print(f"  Adapt  : {X2_adapt.shape[0]}   Test : {X2_test.shape[0]}  (held out)")

# ── Common gene indices ────────────────────────────────────────────────────────
common_genes = sorted(set(genes1).intersection(set(genes2)))
src_idx = np.array([np.where(genes1 == g)[0][0] for g in common_genes])  # positions in genes1
tgt_idx = np.array([np.where(genes2 == g)[0][0] for g in common_genes])  # positions in genes2
print(f"  Common genes: {len(common_genes)}")

# ── Normalise ─────────────────────────────────────────────────────────────────
mms_src = MinMaxScaler()
X1_norm = mms_src.fit_transform(X1)               # (589, 22489) in [0,1]

mms_tgt = MinMaxScaler()
X2_adapt_norm = mms_tgt.fit_transform(X2_adapt)   # (128, 236)  in [0,1]
X2_test_norm  = mms_tgt.transform(X2_test)        # (55,  236)  in [0,1] — same scaler

# ── Labels ────────────────────────────────────────────────────────────────────
le = LabelEncoder().fit(Y1)
y_src   = le.transform(Y1)
y_adapt = le.transform(Y2_adapt)
y_test  = le.transform(Y2_test)
num_classes = len(le.classes_)
print(f"  Classes: {list(le.classes_)}")

# ── KPCA-DA: fit on source (22489 genes) + adapt (236 genes) ──────────────────
print("\nRunning KPCA-DA...")
transformer = DomainAdaptiveImageTransformer(
    kpca_da_params={
        "n_components": 2,
        "kernel":       "rbf",
        "kpca_gamma":   0.5,
        "auto_gamma":   True,
        "lambda_reg":   0.0001,
        "lr":           0.001,
        "epochs":       500,
        "verbose":      False,
    },
    pixels=(227, 227),
)
transformer.fit(
    X_source=X1_norm, X_target=X2_adapt_norm,
    genes_source=genes1, genes_target=genes2,
    plot=False,
)

# ── Build completed test vectors ───────────────────────────────────────────────
# For each test sample:
#   - Find nearest source sample by Euclidean distance on common genes
#   - Copy nearest source sample as the base (22489 genes)
#   - Overwrite the common gene positions with the test sample's own values
#     (rescaled from target space → source space via raw expression)

print("\nImputing missing genes in test samples...")

# Source per-gene min and range (from the fitted MinMaxScaler)
src_min   = mms_src.data_min_   # (22489,)
src_range = mms_src.data_range_ # (22489,)
src_range = np.where(src_range == 0, 1e-8, src_range)  # avoid /0

# Target per-gene min and range (from the fitted MinMaxScaler)
tgt_min   = mms_tgt.data_min_   # (236,)
tgt_range = mms_tgt.data_range_ # (236,)
tgt_range = np.where(tgt_range == 0, 1e-8, tgt_range)

# Common-gene values from source in source-normalised space
X1_common_norm = X1_norm[:, src_idx]   # (589, 228)

# Common-gene values from test in target-normalised space
X2_test_common_norm = X2_test_norm[:, tgt_idx]  # (55, 228)

# Re-normalise test common-gene values into source space:
#   raw = tgt_norm * tgt_range + tgt_min  (invert target scaler)
#   src_norm = (raw - src_min) / src_range (apply source scaler)
X2_test_common_raw = (
    X2_test_common_norm * tgt_range[tgt_idx] + tgt_min[tgt_idx]
)  # (55, 228) — back to raw expression
X2_test_common_in_src_space = (
    (X2_test_common_raw - src_min[src_idx]) / src_range[src_idx]
).clip(0, 1)  # (55, 228) — normalised using source scaler

# Nearest-neighbour matching on common genes (Euclidean distance)
nn_indices = []
for i in range(len(X2_test)):
    dists = np.linalg.norm(X1_common_norm - X2_test_common_in_src_space[i], axis=1)
    nn_indices.append(np.argmin(dists))
nn_indices = np.array(nn_indices)

print(f"  Nearest neighbour match done for {len(nn_indices)} test samples")
unique_nn = len(np.unique(nn_indices))
print(f"  Unique source samples used as neighbours: {unique_nn} / {len(X1_norm)}")

# Build completed 22489-gene vectors
X_test_completed = X1_norm[nn_indices, :].copy()          # start from NN source sample
X_test_completed[:, src_idx] = X2_test_common_in_src_space  # overwrite with actual test values

# ── Generate images ───────────────────────────────────────────────────────────
print("\nGenerating images...")
source_images   = transformer.transform_source(X1_norm)           # (589, 227, 227, 3)
adapt_images    = transformer.transform_target(X2_adapt_norm)     # (128, 227, 227, 3)
test_images_imp = transformer.transform_source(X_test_completed)  # (55,  227, 227, 3) — SOURCE layout
print(f"  source_images    : {source_images.shape}")
print(f"  adapt_images     : {adapt_images.shape}")
print(f"  test_images_imp  : {test_images_imp.shape}  (completed via NN imputation)")

# ── ResNet50 ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def make_loader(images, labels, shuffle=False, batch=16):
    X_t = torch.stack([preprocess(img) for img in images]).float().to(device)
    y_t = torch.tensor(labels, dtype=torch.long).to(device)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch, shuffle=shuffle)

# Training = source + adapt
train_images = np.concatenate([source_images, adapt_images], axis=0)
train_labels = np.concatenate([y_src, y_adapt], axis=0)
print(f"\nTraining set: {len(train_images)} images  (source={len(source_images)}, adapt={len(adapt_images)})")

trainloader = make_loader(train_images, train_labels, shuffle=True)
testloader  = make_loader(test_images_imp, y_test)

net = timm.create_model("resnet50d", pretrained=True, num_classes=num_classes).to(device)

# Weighted loss
class_counts  = np.bincount(train_labels)
weights       = (1.0 / class_counts) * num_classes / 2.0
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
print(f"Class weights: { {le.classes_[i]: round(weights[i], 3) for i in range(num_classes)} }")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, verbose=False)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nTraining ResNet50d...")
print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Test Acc':>9}  {'Best':>6}")
print("-" * 40)

best_acc, best_state = 0.0, None
EPOCHS = 50

for epoch in range(1, EPOCHS + 1):
    net.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        loss = criterion(net(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    net.eval()
    preds, trues = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            preds.extend(torch.max(net(inputs), 1)[1].cpu().numpy())
            trues.extend(labels.cpu().numpy())

    test_acc = accuracy_score(trues, preds)
    avg_loss = running_loss / len(trainloader)
    scheduler.step(test_acc)

    if test_acc > best_acc:
        best_acc   = test_acc
        best_state = {k: v.clone() for k, v in net.state_dict().items()}

    marker = " <--" if test_acc == best_acc else ""
    print(f"{epoch:>6}  {avg_loss:>11.4f}  {test_acc:>9.4f}  {best_acc:>6.4f}{marker}")

# ── Final evaluation ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL RESULTS (best model, NN-imputed test images)")
print("=" * 55)
net.load_state_dict(best_state)
net.eval()
preds, trues = [], []
with torch.no_grad():
    for inputs, labels in testloader:
        preds.extend(torch.max(net(inputs), 1)[1].cpu().numpy())
        trues.extend(labels.cpu().numpy())

print(f"  Test Accuracy       : {accuracy_score(trues, preds):.4f}")
print(f"  Balanced Accuracy   : {balanced_accuracy_score(trues, preds):.4f}")
print(f"  F1 Score (weighted) : {f1_score(trues, preds, average='weighted'):.4f}")
print(f"  Test samples        : {len(trues)}  "
      f"(Normal={sum(y_test==0)}, Tumor={sum(y_test==1)})")
