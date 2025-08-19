import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
import sys
sys.path.insert(1,'C:\Zmyfiles\codes\research\pyDeepInsight_SRT\pyDeepInsight_SRT\examples\Scripts')

#import dataset
#SOURCE
expr_file1= r"C:\Zmyfiles\codes\research\pyDeepInsight_SRT\pyDeepInsight_SRT\examples\data\TvsN\TCGA_BRCA_NvT.csv"
expr1 = pd.read_csv(expr_file1, encoding='utf-8', engine='python')
expr1=expr1.transpose()
expr1.columns=expr1.iloc[0]
X1 = expr1.iloc[1:,1:-1].values
X_S=X1.astype(float)
X_S=X_S.T

#TARGET
expr_file2= r"C:\Zmyfiles\codes\research\pyDeepInsight_SRT\pyDeepInsight_SRT\examples\Data\TvsN\GSE45498_NvT.csv"
expr2 = pd.read_csv(expr_file2, encoding='utf-8', engine='python')
expr2=expr2.transpose()
expr2.columns=expr2.iloc[0]
X2 = expr2.iloc[1:,1:-1].values
X_T=X2.astype(float)
X_T=X_T.T

###Normalize the source and target dataset
scaler = StandardScaler()
X_S = scaler.fit_transform(X_S)
X_T = scaler.fit_transform(X_T)

#extract common genes
common_indices_X_S = []
common_indices_X_T = []
for i, row in enumerate(X_S):
    for j, candidate in enumerate(X_T):
        if np.array_equal(row, candidate):
            common_indices_X_S.append(i)
            common_indices_X_T.append(j)

common_indices_X_S = np.array(common_indices_X_S)
common_indices_X_T = np.array(common_indices_X_T)


# Example data
kpca = KernelPCA(n_components=2, kernel='rbf')
Z_s = kpca.fit_transform(X_S)
Z_t = kpca.fit_transform(X_T)   # Target domain features
X_t = X_T.T                     # Original target domain data

Z_s = torch.tensor(Z_s, dtype=torch.float32)
Z_t = torch.tensor(Z_t, dtype=torch.float32, requires_grad=True)  # Make Z_t trainable
X_t = torch.tensor(X_t, dtype=torch.float32)

# Extract common points from Z1 and Z2
Zs_common = Z_s[common_indices_X_S]
Zt_common = Z_t[common_indices_X_T]
print("Zs_common : ",Zs_common)
print("Zt_common : ",Zt_common)

# Define the RBF kernel
def rbf_kernel(x, y, gamma=1.0):
    pairwise_distances = torch.cdist(x, y) ** 2
    return torch.exp(-gamma * pairwise_distances)

# Define the MMD loss
def mmd_loss(Zs_common, Zt_common, gamma=1.0):
    # if(torch.isnan(Zs_common) or torch.isinf(Zs_common)):
    #     print("NaN Error in Source")
    #     exit()

    # if(torch.isnan(Zt_common) or torch.isinf(Zt_common)):
    #     print("NaN Error in Target")
    #     exit()

    K_ss = rbf_kernel(Zs_common, Zs_common, gamma)
    K_tt = rbf_kernel(Zt_common, Zt_common, gamma)
    K_st = rbf_kernel(Zs_common, Zt_common, gamma)
    
    mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    return mmd

# Define the regularization term
def regularization_term(Z_t, X_t, gamma=1.0):
    #print(Z_t.shape, X_t.shape)
    K_t = rbf_kernel(X_t.T, X_t.T, gamma)
    #print(K_t.shape)
    A = Z_t.T @ K_t @ Z_t
    trace_A = torch.trace(A)
    return trace_A


# Hyperparameters
lambda_reg = 0.1  # Regularization strength
gamma = 1.0       # Kernel parameter

# Optimizer
optimizer = optim.SGD([Z_t], lr=0.01)

# Training loop
for epoch in range(100):
    Z_t = Z_t.detach().requires_grad_(True)
    optimizer.zero_grad()
    
    # Compute MMD loss
    Zt_common = Z_t[common_indices_X_T]
    mmd = mmd_loss(Zs_common, Zt_common, gamma)
    
    # Compute regularization term
    reg = regularization_term(Z_t, X_t, gamma)
    
    # Total loss
    loss = mmd + lambda_reg * reg
    
    # Backpropagate and update Z_t
    loss.backward(retain_graph=True)
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, MMD: {mmd.item():.4f}, Reg: {reg.item():.4f}")

# Final Z_t after optimization
print("Optimized Z_t:", Z_t)