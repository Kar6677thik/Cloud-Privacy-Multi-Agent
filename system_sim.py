import numpy as np
from he_layer import get_he_context
from spectral_core import SpectralClusteringCore

class KGC:
    """Key Generation Center"""
    def __init__(self, he_type="mock"):
        self.he = get_he_context(he_type)

    def initialize_system(self):
        pk, sk = self.he.keygen()
        return self.he, pk, sk


class DataOwner:
    """
    Represents a user with a private dataset.
    """
    def __init__(self, user_id, data):
        """
        data: numpy array of shape (n_samples, n_features)
        """
        self.user_id = user_id
        self.data = data
        
    def encrypt_and_upload(self, he_context):
        """
        Encrypts rows of data and returns them.
        """
        encrypted_rows = []
        for row in self.data:
            # Encrypt each row vector
            encrypted_rows.append(he_context.encrypt(row))
        return encrypted_rows


class KeyServer:
    """
    Trusted party holding the secret key.
    In this prototype, it shares the HE context logic.
    """
    def __init__(self, sk):
        self.sk = sk


class CiphertextServer:
    """
    Cloud server that performs computations on encrypted data.
    """
    def __init__(self, he_context):
        self.he = he_context
        self.spectral = SpectralClusteringCore(he_context)
        self.encrypted_pool = []
        
    def collect_data(self, encrypted_datasets):
        """
        Aggregates data from multiple users.
        encrypted_datasets: list of lists of Ciphertexts
        """
        self.encrypted_pool = []
        for ds in encrypted_datasets:
            self.encrypted_pool.extend(ds)
            
    def compute_clustering(self, k, sigma, epsilon=None):
        """
        Executes the spectral clustering protocol.
        """
        if not self.encrypted_pool:
            raise ValueError("No data collected in Ciphertext Server.")
            
        # Compute Affinity Matrix first
        W_enc = self.spectral.compute_encrypted_affinity_matrix(self.encrypted_pool, sigma)
        return self.spectral.solve_clustering(W_enc, k, epsilon=epsilon)


def simulate_multi_user_system(data_owners_data, k, sigma, he_type="mock", epsilon=None):
    """
    Simulates the full flow.
    
    Args:
        data_owners_data: Dict or List of datasets (arrays) from different users.
                          Example: {"User1": np.array(...), "User2": ...}
        k: Number of clusters
        sigma: Kernel sigma
        he_type: "mock" or "tenseal"
        epsilon: Differential Privacy Budget (float or None)
        
    Returns:
        results: Dictionary containing labels, embedding, W, L, metrics
    """
    from metrics import calculate_metrics # Local import to avoid circular dependency if any
    
    # 1. Setup Phase (KGC)
    kgc = KGC(he_type=he_type)
    he, pk, sk = kgc.initialize_system()
    
    # 2. Data Upload Phase (Data Owners)
    # Instantiate CS1
    cs1 = CiphertextServer(he)
    
    encrypted_payloads = []
    
    # Handle input format (dict or list)
    if isinstance(data_owners_data, dict):
        items = data_owners_data.items()
    else:
        # enumerate list
        items = [(f"User_{i}", data) for i, data in enumerate(data_owners_data)]
        
    total_samples = 0
    ground_truth_owners = [] # Track which owner each sample belongs to for validataion/viz if needed
    
    for uid, data in items:
        owner = DataOwner(uid, data)
        enc_data = owner.encrypt_and_upload(he)
        encrypted_payloads.append(enc_data)
        
        n = len(data)
        # Map user ID to integer for ARI calculation if needed, or keep string
        ground_truth_owners.extend([uid] * n)
        total_samples += n
        
    # 3. Aggregation (CS1)
    cs1.collect_data(encrypted_payloads)
    
    # 4. Computation (CS1 + CS2 interaction simulated)
    labels, Y, W, L = cs1.compute_clustering(k, sigma, epsilon=epsilon)
    
    # Calculate Quality Metrics
    # Need full data X? We have Y (embedding) and raw data (aggregated_data not easily available here without concat)
    # Ideally metrics like Silhouette use the Embedding Y since that's what was clustered,
    # OR the original space. Spectral Clustering optimizes the Embedding, so evaluating Y is fair.
    # Evaluating original X evaluates if Spectral Clustering preserved original structure.
    
    metrics = calculate_metrics(Y, labels, true_labels=ground_truth_owners)
    
    return {
        "labels": labels,
        "Y": Y,
        "W": W,
        "L": L,
        "owner_labels": ground_truth_owners,
        "metrics": metrics
    }
