import numpy as np
import copy

class Ciphertext:
    """
    A mock ciphertext wrapper.
    In a real system, this would hold encrypted polynomial coefficients (CKKS).
    Here, it wraps a NumPy array or scalar and adds 'noise' metadata
    to simulate the HE experience.
    """
    def __init__(self, data, is_encrypted=True):
        self.data = np.array(data)
        self.is_encrypted = is_encrypted
        self.noise_budget = 100.0 if is_encrypted else float('inf')

    def __repr__(self):
        shape_str = str(self.data.shape)
        return f"<Ciphertext shape={shape_str}, encrypted={self.is_encrypted}>"


class HEContext:
    """
    Mock HE Context to manage keys and operations.
    Simulates CKKS operations.
    """
    def __init__(self):
        self.public_key = None
        self.secret_key = None
        self.relin_keys = None

    def keygen(self):
        """Generates mock keys."""
        self.public_key = "pk_mock"
        self.secret_key = "sk_mock"
        self.relin_keys = "rk_mock"
        return self.public_key, self.secret_key

    def encrypt(self, x):
        """Encrypts data (wraps it in Ciphertext)."""
        return Ciphertext(x, is_encrypted=True)

    def decrypt(self, ct: Ciphertext):
        """Decrypts data (unwraps it). Checks secret key in real life."""
        if not isinstance(ct, Ciphertext):
            raise ValueError("Input must be a Ciphertext")
        # In a real system, we'd check if we have the secret key here
        return ct.data

    def add(self, ct1, ct2):
        """Homomorphic addition."""
        val1 = ct1.data if isinstance(ct1, Ciphertext) else ct1
        val2 = ct2.data if isinstance(ct2, Ciphertext) else ct2
        return Ciphertext(val1 + val2)

    def sub(self, ct1, ct2):
        """Homomorphic subtraction."""
        val1 = ct1.data if isinstance(ct1, Ciphertext) else ct1
        val2 = ct2.data if isinstance(ct2, Ciphertext) else ct2
        return Ciphertext(val1 - val2)

    def mul(self, ct1, ct2):
        """Homomorphic multiplication."""
        val1 = ct1.data if isinstance(ct1, Ciphertext) else ct1
        val2 = ct2.data if isinstance(ct2, Ciphertext) else ct2
        return Ciphertext(val1 * val2)

    def scalar_mul(self, a, ct):
        """Multiplication by plain scalar."""
        if isinstance(ct, Ciphertext):
            return Ciphertext(a * ct.data)
        else:
            return Ciphertext(a * ct)

    def l2_norm_sq(self, ct_a, ct_b):
        """
        Computes ||a - b||^2 homomorphically.
        Result is a Ciphertext scalar.
        """
        diff = self.sub(ct_a, ct_b)
        # Sum of squares of elements
        # In CKKS, this is a sequence of mul and add (relinearization needed usually)
        squared_diff = diff.data ** 2
        sum_sq = np.sum(squared_diff)
        return Ciphertext(sum_sq)

    def gaussian_kernel(self, ct_a, ct_b, sigma):
        """
        Computes exp(-||a - b||^2 / (2 * sigma^2)).
        Note: True HE exponentiation is hard. We simulate it here
        or assume a polynomial approximation exists.
        For this prototype, we compute it directly on underlying data
        to keep the 'spectral' part working robustly.
        """
        norm_sq = self.decrypt(self.l2_norm_sq(ct_a, ct_b))
        val = np.exp(-norm_sq / (2 * sigma**2))
        return Ciphertext(val)
        
    def negative_half_power(self, ct):
        """
        Computes x^(-1/2). Used for D^(-1/2).
        """
        # decrypt to compute inverse sqrt
        val = self.decrypt(ct)
        # Handle division by zero or negative gracefully for prototype
        with np.errstate(divide='ignore'):
             res = 1.0 / np.sqrt(val)
        
        if np.isscalar(res):
            if np.isinf(res):
                res = 0.0
        else:
            res[np.isinf(res)] = 0
            
        return Ciphertext(res)

    def exp(self, ct):
        """Homomorphic exponential (simulated)."""
        val = self.decrypt(ct)
        return Ciphertext(np.exp(val))
