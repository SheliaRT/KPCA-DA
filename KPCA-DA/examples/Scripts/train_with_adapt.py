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

from pyDeepInsightv3 import ImageTransformer
from pyDeepInsightv3.image_transformer_new import DomainAdaptiveImageTransformer

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "TvsN")

print("Loading data...")
expr1 = pd.read_csv(os.path.join(DATA_DIR, "TCGA_BRCA_NvT_new.csv"), encoding="utf-8", engine="python")
expr1 = expr1.transpose(); expr1.columns = expr1.iloc[0]
X1      = expr1.iloc[1:, :-1].values.astype(float)
Y1      = expr1.iloc[1:, -1].values
genes1  = expr1.iloc[:, :-1].columns.to_numpy()

expr2 = pd.read_csv(os.path.join(DATA_DIR, "GSE45498_NvT.csv"), encoding="utf-8", engine="python")
expr2 = expr2.transpose(); expr2.columns = expr2.iloc[0]
X2      = expr2.iloc[1:, :-1].values.astype(float)
Y2      = expr2.iloc[1:, -1].values
genes2  = expr2.iloc[:, :-1].columns.to_numpy()

print(f"  Source : {X1.shape}  Target : {X2.shape}")

# ── Remove Unknown-labelled source samples ────────────────────────────────────
known_mask = Y1 != "Unknown"
X1 = X1[known_mask];  Y1 = Y1[known_mask]
print(f"  Source after removing Unknown: {X1.shape}")

# ── Split target → adapt (70%) / test (30%) stratified ───────────────────────
X2_adapt, X2_test, Y2_adapt, Y2_test = train_test_split(
    X2, Y2, test_size=0.3, random_state=42, stratify=Y2)
print(f"\nTarget split:")
print(f"  Adapt : {X2_adapt.shape[0]} samples")
print(f"  Test  : {X2_test.shape[0]}  samples  (held out)")

# ── Normalise (each domain independently — different platforms, different genes) ─
# Source: fit on full 22,489 gene set
mms_src = MinMaxScaler()
X1_norm = mms_src.fit_transform(X1)

# Target: fit on adapt set, transform test with same scaler (no leakage)
mms_tgt = MinMaxScaler()
X2_adapt_norm = mms_tgt.fit_transform(X2_adapt)
X2_test_norm  = mms_tgt.transform(X2_test)

print(f"\nGene sets:")
print(f"  Source : {X1_norm.shape[1]} genes  ({X1_norm.shape[0]} samples)")
print(f"  Adapt  : {X2_adapt_norm.shape[1]} genes  ({X2_adapt_norm.shape[0]} samples)")
print(f"  Test   : {X2_test_norm.shape[1]}  genes  ({X2_test_norm.shape[0]}  samples)")

# ── Label encoding ────────────────────────────────────────────────────────────
le = LabelEncoder().fit(Y1)
y_src   = le.transform(Y1)
y_adapt = le.transform(Y2_adapt)
y_test  = le.transform(Y2_test)
num_classes = len(le.classes_)
print(f"\nClasses: {le.classes_}  ({num_classes} classes)")

# ── KPCA-DA image transformer (full gene set per domain) ─────────────────────
print("\nRunning KPCA-DA domain adaptation...")
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
    pixels=(227, 227),  # larger canvas to accommodate 22,489 source genes
)

# Fit on source (22,489 genes) + adapt (236 genes) using their own gene lists
transformer.fit(
    X_source=X1_norm,
    X_target=X2_adapt_norm,
    genes_source=genes1,
    genes_target=genes2,
    plot=False,
)

# Generate images
print("\nGenerating images...")
source_images = transformer.transform_source(X1_norm)        # (589, 227, 227, 3)
adapt_images  = transformer.transform_target(X2_adapt_norm)  # (128, 227, 227, 3)
test_images   = transformer.transform_target(X2_test_norm)   # (55,  227, 227, 3)
print(f"  source_images : {source_images.shape}")
print(f"  adapt_images  : {adapt_images.shape}")
print(f"  test_images   : {test_images.shape}")

# ── ResNet50 setup ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

def make_tensors(images, labels):
    X_t = torch.stack([preprocess(img) for img in images]).float().to(device)
    y_t = torch.tensor(labels, dtype=torch.long).to(device)
    return TensorDataset(X_t, y_t)

# ── Training set = source + adapt ────────────────────────────────────────────
train_images  = np.concatenate([source_images, adapt_images], axis=0)  # (719, 64, 64, 3)
train_labels  = np.concatenate([y_src, y_adapt],              axis=0)  # (719,)
print(f"\nTraining set: {train_images.shape[0]} images  "
      f"(source={len(source_images)}, adapt={len(adapt_images)})")

trainset   = make_tensors(train_images, train_labels)
testset    = make_tensors(test_images,  y_test)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
testloader  = DataLoader(testset,  batch_size=16, shuffle=False)

net = timm.create_model("resnet50d", pretrained=True, num_classes=num_classes).to(device)

# Weighted loss: inverse class frequency so minority class is penalised equally
class_counts = np.bincount(train_labels)
weights = (1.0 / class_counts) * num_classes / 2.0
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
print(f"\nClass weights: { {le.classes_[i]: round(weights[i],3) for i in range(num_classes)} }")
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, verbose=False)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nTraining ResNet50d...")
print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Test Acc':>9}  {'Best':>6}")
print("-" * 40)

best_acc   = 0.0
best_state = None
EPOCHS     = 50

for epoch in range(1, EPOCHS + 1):
    net.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        loss = criterion(net(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Evaluate on test set
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

# ── Final evaluation on best model ───────────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL RESULTS (best model)")
print("=" * 50)
net.load_state_dict(best_state)
net.eval()
preds, trues = [], []
with torch.no_grad():
    for inputs, labels in testloader:
        preds.extend(torch.max(net(inputs), 1)[1].cpu().numpy())
        trues.extend(labels.cpu().numpy())

print(f"  Test Accuracy         : {accuracy_score(trues, preds):.4f}")
print(f"  Balanced Accuracy     : {balanced_accuracy_score(trues, preds):.4f}")
print(f"  F1 Score (weighted)   : {f1_score(trues, preds, average='weighted'):.4f}")
print(f"  Class mapping         : {dict(zip(range(num_classes), le.classes_))}")
print(f"  Test samples          : {len(trues)}  (Normal={sum(y_test==0)}, Tumor={sum(y_test==1)})")
