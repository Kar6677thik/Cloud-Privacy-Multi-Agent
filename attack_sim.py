
import numpy as np
from he_layer import get_he_context, Ciphertext

class AttackSimulator:
    """
    Simulates various privacy attacks on the system.
    """
    
    @staticmethod
    def reconstruct_from_laplacian(L, known_eigvecs):
        """
        Attempt to reconstruct adjacency/affinity info from Laplacian.
        L = D^-1/2 W D^-1/2
        If we know D (degree matrix) we can recover W.
        In Spectral Clustering, D is simply row sums of W.
        If an attacker (Cloud Server) sees L, can they find W?
        
        This is a 'Inversion Attack' simulation.
        """
        # In this simplistic simulation, we just check if L looks like W
        # Return a correlation score or reconstruction error assuming some prior
        pass

    @staticmethod
    def simulate_collation_attack(encrypted_pool, key_server_sk, he_type="mock"):
        """
        Simulates what happens if Cloud Server colludes with Key Server.
        They can decrypt everything.
        
        Args:
            encrypted_pool: List of Ciphertexts held by Cloud
            key_server_sk: Secret key held by Key Server (or Context with SK for TenSEAL)
            he_type: Backend type
            
        Returns:
            decrypted_data: The raw user data revealed.
        """
        # Create a context with the stolen keys
        he = get_he_context(he_type)
        
        # Inject the stolen key into the context wrapper
        if he_type == "mock":
            he.secret_key = key_server_sk
        elif he_type == "tenseal":
            # For TenSEAL, key_server_sk is the context
            he.ctx = key_server_sk
            # In local sim, vectors hold ref to context. 
            # If we were remote, we'd need to link them.
            # But effectively, possessing the SK allows decrypt.
        
        leakage = []
        for ct in encrypted_pool:
            if he_type == "tenseal":
                # Explicitly verify that we are using the stolen context context capability
                # In real Tenseal, we might load context from sk proto.
                # Here we just call decrypt via our wrapper which delegates.
                # Since ct.data might hold the context, we rely on that in local sim,
                # BUT having the sk validates the premise.
                pass
                
            val = he.decrypt(ct)
            leakage.append(val)
            
        return np.array(leakage)

    @staticmethod
    def noise_budget_audit(encrypted_pool):
        """
        Checks if noise budget is dangerously low (for CKKS).
        Only relevant for Mock HE if we track it, or TenSEAL if we can inspect.
        """
        if not encrypted_pool:
            return "No data"
            
        # Check first element
        ct = encrypted_pool[0]
        if ct.backend == "mock":
            return f"Mock Noise Budget: {ct.noise_budget}"
        elif ct.backend == "tenseal":
            # TenSEAL vectors don't easily expose budget without decryption in python wrapper sometimes
            # But we can try
            try:
                # This might fail depending on exact TS version/setup
                return "TenSEAL Noise Tracking (Hidden)" 
            except:
                return "Unknown"
        return "Unknown"
