import numpy as np

def rbf_kernel(X, gamma=1.0):
    """
    Compute the RBF (Gaussian) kernel matrix for a dataset X.
    
    Parameters:
    X (numpy.ndarray): Data matrix of shape (n_samples, n_features).
    gamma (float): Parameter for the RBF kernel.
    
    Returns:
    numpy.ndarray: Kernel matrix of shape (n_samples, n_samples).
    """
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    K = np.exp(-gamma * sq_dists)
    return K

def center_kernel_matrix(K):
    """
    Center the kernel matrix K.
    
    Parameters:
    K (numpy.ndarray): Kernel matrix of shape (n_samples, n_samples).
    
    Returns:
    numpy.ndarray: Centered kernel matrix.
    """
    n_samples = K.shape[0]
    one_n = np.ones((n_samples, n_samples)) / n_samples
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_centered

def kernel_pca(X, n_components=2, gamma=1.0):
    """
    Perform Kernel Principal Component Analysis (kPCA) on the dataset X.
    
    Parameters:
    X (numpy.ndarray): Data matrix of shape (n_samples, n_features).
    n_components (int): Number of principal components to compute.
    gamma (float): Parameter for the RBF kernel.
    
    Returns:
    numpy.ndarray: Transformed data matrix of shape (n_samples, n_components).
    """
    # Step 1: Compute the kernel (RBF) matrix
    K = rbf_kernel(X, gamma=gamma)

    # Step 2: Center the kernel matrix
    K_centered = center_kernel_matrix(K)

    # Step 3: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Select the top n_components eigenvectors and transform the data
    X_transformed = np.dot(K_centered, eigenvectors[:, :n_components])

    return X_transformed

# Example usage
# if __name__ == "__main__":
#     # Sample dataset (e.g., 4 samples, 2 features)
#     X = np.array([[1, 2,3, 4], [5, 6,7, 8],[9,10,11,12]])

#     # Perform kPCA with 2 components and gamma=0.5
#     X_kpca = kernel_pca(X, n_components=3, gamma=0.5)
#     print("Transformed data using Kernel PCA:\n", X_kpca)
