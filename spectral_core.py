import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from he_layer import HEContext, Ciphertext

class SpectralClusteringCore:
    def __init__(self, he_context: HEContext):
        self.he = he_context

    def compute_encrypted_affinity_matrix(self, encrypted_data, sigma):
        """
        Computes the Gaussian affinity matrix W homomorphically.
        W_ij = exp( -||x_i - x_j||^2 / (2*sigma^2) )
        
        Args:
            encrypted_data: List or array of Ciphertext vectors (rows of data).
            sigma: Gaussian kernel parameter.
            
        Returns:
            W: 2D array of Ciphertexts (N x N)
        """
        n_samples = len(encrypted_data)
        W = np.empty((n_samples, n_samples), dtype=object)
        
        # In a real secure protocol, we might compute only upper triangular and mirror,
        # but here we compute fully for simplicity or optimization as needed.
        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    # Self-similarity is usually 0 distance -> exp(0) = 1
                    # We can directly encrypt 1.0 or compute it.
                    # exp(-0) = 1.
                    W[i, j] = self.he.encrypt(1.0)
                else:
                    # Compute similarity
                    sim = self.he.gaussian_kernel(encrypted_data[i], encrypted_data[j], sigma)
                    W[i, j] = sim
                    W[j, i] = sim # Symmetric
        
        return W

    def compute_encrypted_laplacian(self, W_enc):
        """
        Computes the normalized Laplacian L_sym = D^(-1/2) * W * D^(-1/2) (usually I - this, but standard varies)
        Commonly for clustering: L = D^(-1/2) * W * D^(-1/2) then look at eigenvectors of I-L or L depending on convention.
        
        We will use the normalized symmetric Laplacian formula:
        L_sym = I - D^(-1/2) W D^(-1/2)
        
        However, pure spectral clustering usually takes eigenvectors of L_sym corresponding to smallest eigenvalues,
        OR eigenvectors of D^(-1/2) W D^(-1/2) corresponding to LARGEST eigenvalues (equivalent to smallest of I - ...).
        
        Let's compute M = D^(-1/2) W D^(-1/2) and then later we can either eigensolve M or I-M.
        
        Args:
            W_enc: Encrypted Affinity Matrix (N x N Ciphertexts)
            
        Returns:
            L_enc: Encrypted matrix M = D^(-1/2) W D^(-1/2)
        """
        n = W_enc.shape[0]
        
        # 1. Compute Degree Matrix Row Sums (Encrypted)
        D_diag = []
        for i in range(n):
            # sum over j of W_ij
            row_sum = W_enc[i, 0]
            for j in range(1, n):
                row_sum = self.he.add(row_sum, W_enc[i, j])
            D_diag.append(row_sum)
            
        # 2. Compute D^(-1/2) (Encrypted)
        D_inv_sqrt = []
        for d in D_diag:
            D_inv_sqrt.append(self.he.negative_half_power(d))
            
        # 3. Compute M = D^(-1/2) * W * D^(-1/2)
        # Element M_ij = W_ij * D_ii^(-1/2) * D_jj^(-1/2)
        M_enc = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                # Product of 3 ciphertexts.
                # In real CKKS this raises depth significantly (depth 2 multiplication).
                # term1 = w_ij * d_inv_sqrt_i
                term1 = self.he.mul(W_enc[i, j], D_inv_sqrt[i])
                # result = term1 * d_inv_sqrt_j
                M_enc[i, j] = self.he.mul(term1, D_inv_sqrt[j])
                
        return M_enc

    def solve_clustering(self, W_enc, k, epsilon=None):
        """
        The full pipeline from Encrypted Affinity Matrix W_enc.
        
        1. Compute L (Encrypted)
        2. Decrypt L (Simulated "Client/Server" cooperation or final step)
        3. Eigendecomposition
        4. Differential Privacy Noise Addition (Optional)
        5. K-Means
        
        Args:
            W_enc: Encrypted W
            k: number of clusters
            epsilon: Privacy Budget (None or float). If set, adds Laplace noise.
            
        Returns:
            labels: Cluster labels
            Y: Spectral embedding (potentially noisy)
            W_plain: Decrypted W (for vis)
            L_plain: Decrypted L (for vis)
        """
        # 1. Compute Matrix M (part of Laplacian) encrypted
        # Using M = D^-1/2 W D^-1/2.
        # Eigenvectors of M corresponding to Largest eigenvalues are equivalent to
        # Smallest eigenvalues of L = I - M.
        M_enc = self.compute_encrypted_laplacian(W_enc)
        
        # 2. Decrypt matrices for Eigendecomposition (Simulated)
        n = W_enc.shape[0]
        W_plain = np.zeros((n, n))
        M_plain = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                W_plain[i, j] = self.he.decrypt(W_enc[i, j])
                M_plain[i, j] = self.he.decrypt(M_enc[i, j])
                
        # 3. Eigendecomposition
        # We want k eigenvectors associated with the k LARGEST eigenvalues of M
        # (Since M relates to association, L relates to cost. Max assoc ~ Min cut)
        # scipy.linalg.eigh returns eigenvalues in ascending order.
        # So we take the LAST k columns.
        eigvals, eigvecs = eigh(M_plain)
        
        # Select k largest
        # Indices of k largest
        idx = np.argsort(eigvals)[::-1][:k]
        Y = eigvecs[:, idx]
        
        # Normalize rows of Y to unit length (optional but standard in Normalized Spectral Clustering - Ng, Jordan, Weiss)
        rows_norm = np.linalg.norm(Y, axis=1, keepdims=True)
        # Avoid division by zero
        rows_norm[rows_norm == 0] = 1
        Y_norm = Y / rows_norm
        
        # 4. Differential Privacy Layer (Laplace Mechanism)
        if epsilon is not None and epsilon > 0:
            # Sensitivity Analysis (Simplified):
            # Eigenvectors lie on the unit hyper-sphere (row norm 1).
            # A rough sensitivity upper bound could be modeled.
            # For prototype demonstration: scale = 0.1 / epsilon ensures obvious trade-off.
            # Epsilon = 0.1 -> Scale = 1.0 (High noise)
            # Epsilon = 10.0 -> Scale = 0.01 (Low noise)
            scale = 0.5 / epsilon 
            noise = np.random.laplace(0, scale, Y_norm.shape)
            Y_norm = Y_norm + noise
            # Re-normalize might be needed or preferred by K-Means, but raw noisy data is also fine to show displacement.
            
        # 5. K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(Y_norm)
        
        return labels, Y_norm, W_plain, M_plain
