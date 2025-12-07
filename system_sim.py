import numpy as np
from he_layer import HEContext
from spectral_core import SpectralClusteringCore

class KGC:
    """Key Generation Center"""
    def __init__(self):
        self.he = HEContext()

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
            # In a real system, we might encrypt feature-wise or packed.
            # Here we wrap the whole vector in one "Ciphertext" if our operations support vectors,
            # OR list of Ciphertexts.
            # Our he_layer.Ciphertext wraps numpy arrays, so we can wrap the whole row vector.
            # This is efficient for vector operations if supported.
            # spectral_core expects gaussian_kernel to work on these.
            # existing gaussian_kernel takes ct_a, ct_b.
            # he_layer.l2_norm_sq computes ||a-b||^2.
            # If a and b are vector Ciphertexts (wrapping arrays), 
            # sub(a, b) -> Ciphertext(diff_vector)
            # data**2 -> square of elements
            # sum -> scalar sum.
            # YES, he_layer supports vector wrapping.
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
            
    def compute_clustering(self, k, sigma):
        """
        Executes the spectral clustering protocol.
        """
        if not self.encrypted_pool:
            raise ValueError("No data collected in Ciphertext Server.")
            
        # Compute Affinity Matrix first
        W_enc = self.spectral.compute_encrypted_affinity_matrix(self.encrypted_pool, sigma)
        return self.spectral.solve_clustering(W_enc, k)


def simulate_multi_user_system(data_owners_data, k, sigma):
    """
    Simulates the full flow.
    
    Args:
        data_owners_data: Dict or List of datasets (arrays) from different users.
                          Example: {"User1": np.array(...), "User2": ...}
        k: Number of clusters
        sigma: Kernel sigma
        
    Returns:
        results: Dictionary containing labels, embedding, W, L
    """
    # 1. Setup Phase (KGC)
    kgc = KGC()
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
        ground_truth_owners.extend([uid] * n)
        total_samples += n
        
    # 3. Aggregation (CS1)
    cs1.collect_data(encrypted_payloads)
    
    # 4. Computation (CS1 + CS2 interaction simulated)
    labels, Y, W, L = cs1.compute_clustering(k, sigma)
    
    return {
        "labels": labels,
        "Y": Y,
        "W": W,
        "L": L,
        "owner_labels": ground_truth_owners
    }
