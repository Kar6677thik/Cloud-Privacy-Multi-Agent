import numpy as np
import copy

class Ciphertext:
    """
    A wrapper class for encrypted data.
    It can hold either a Mock ciphertext (numpy array with metadata)
    or a Real ciphertext (TenSEAL object).
    """
    def __init__(self, data, is_encrypted=True, backend="mock"):
        self.backend = backend
        self.is_encrypted = is_encrypted
        
        if backend == "mock":
            self.data = np.array(data)
            self.noise_budget = 100.0 if is_encrypted else float('inf')
        elif backend == "tenseal":
            # For TenSEAL, 'data' should already be a CKKSVector or similar object
            self.data = data 
            self.noise_budget = None # Managed internally by TenSEAL

    def __repr__(self):
        if self.backend == "mock":
            shape_str = str(self.data.shape)
            return f"<Ciphertext(Mock) shape={shape_str}, encrypted={self.is_encrypted}>"
        else:
            return f"<Ciphertext(TenSEAL) encrypted={self.is_encrypted}>"


class HEContext:
    """
    Abstract Base Class / Interface for HE Contexts.
    """
    def keygen(self): raise NotImplementedError
    def encrypt(self, x): raise NotImplementedError
    def decrypt(self, ct): raise NotImplementedError
    def add(self, ct1, ct2): raise NotImplementedError
    def sub(self, ct1, ct2): raise NotImplementedError
    def mul(self, ct1, ct2): raise NotImplementedError
    def scalar_mul(self, a, ct): raise NotImplementedError
    def l2_norm_sq(self, ct_a, ct_b): raise NotImplementedError
    def gaussian_kernel(self, ct_a, ct_b, sigma): raise NotImplementedError
    def negative_half_power(self, ct): raise NotImplementedError
    def exp(self, ct): raise NotImplementedError


class MockHEContext(HEContext):
    """
    Mock HE Context to manage keys and operations.
    Simulates CKKS operations using NumPy.
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
        return Ciphertext(x, is_encrypted=True, backend="mock")

    def decrypt(self, ct: Ciphertext):
        """Decrypts data (unwraps it). Checks secret key in real life."""
        if not isinstance(ct, Ciphertext):
             # Ensure we handle recursive decryption if needed or raw values
             return ct 
        if ct.backend != "mock":
            raise ValueError("MockHEContext cannot decrypt non-mock ciphertexts")
            
        return ct.data

    def add(self, ct1, ct2):
        """Homomorphic addition."""
        val1 = ct1.data if isinstance(ct1, Ciphertext) else ct1
        val2 = ct2.data if isinstance(ct2, Ciphertext) else ct2
        return Ciphertext(val1 + val2, backend="mock")

    def sub(self, ct1, ct2):
        """Homomorphic subtraction."""
        val1 = ct1.data if isinstance(ct1, Ciphertext) else ct1
        val2 = ct2.data if isinstance(ct2, Ciphertext) else ct2
        return Ciphertext(val1 - val2, backend="mock")

    def mul(self, ct1, ct2):
        """Homomorphic multiplication."""
        val1 = ct1.data if isinstance(ct1, Ciphertext) else ct1
        val2 = ct2.data if isinstance(ct2, Ciphertext) else ct2
        return Ciphertext(val1 * val2, backend="mock")

    def scalar_mul(self, a, ct):
        """Multiplication by plain scalar."""
        if isinstance(ct, Ciphertext):
            return Ciphertext(a * ct.data, backend="mock")
        else:
            return Ciphertext(a * ct, backend="mock")

    def l2_norm_sq(self, ct_a, ct_b):
        """
        Computes ||a - b||^2 homomorphically.
        Result is a Ciphertext scalar.
        """
        diff = self.sub(ct_a, ct_b)
        squared_diff = diff.data ** 2
        sum_sq = np.sum(squared_diff)
        return Ciphertext(sum_sq, backend="mock")

    def gaussian_kernel(self, ct_a, ct_b, sigma):
        """
        Computes exp(-||a - b||^2 / (2 * sigma^2)).
        """
        norm_sq = self.decrypt(self.l2_norm_sq(ct_a, ct_b))
        val = np.exp(-norm_sq / (2 * sigma**2))
        return Ciphertext(val, backend="mock")
        
    def negative_half_power(self, ct):
        """
        Computes x^(-1/2). Used for D^(-1/2).
        """
        val = self.decrypt(ct)
        with np.errstate(divide='ignore'):
             res = 1.0 / np.sqrt(val)
        
        if np.isscalar(res):
            if np.isinf(res):
                res = 0.0
        else:
            res[np.isinf(res)] = 0
            
        return Ciphertext(res, backend="mock")

    def exp(self, ct):
        """Homomorphic exponential (simulated)."""
        val = self.decrypt(ct)
        return Ciphertext(np.exp(val), backend="mock")


class TenSEALContext(HEContext):
    """
    Real HE Context using TenSEAL (CKKS).
    """
    def __init__(self):
        try:
            import tenseal as ts
            self.ts = ts
        except ImportError:
            raise ImportError("TenSEAL is not installed. Please pip install tenseal.")
            
        self.ctx = None
        
    def keygen(self):
        # Create TenSEAL context
        # Poly moduli degree 16384 for more security and depth capacity
        # Coeff modulus bit sizes: More primes allow for more multiplicative depth
        # The spectral clustering algorithm requires multiple chained multiplications
        # (W * D_inv_sqrt[i] * D_inv_sqrt[j]), so we need sufficient depth
        self.ctx = self.ts.context(
            self.ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 60]  # 5 levels of mult depth
        )
        self.ctx.global_scale = 2**40
        
        # Enable automatic rescaling and relinearization after multiplications
        # This is CRITICAL to prevent "scale out of bounds" errors
        self.ctx.auto_rescale = True
        self.ctx.auto_relin = True
        
        self.ctx.generate_galois_keys()
        self.ctx.generate_relin_keys()
        
        return self.ctx, self.ctx # Public and Secret context are same object in TenSEAL usually for simple use

    def encrypt(self, x):
        # x is expected to be a list or numpy array
        if isinstance(x, np.ndarray):
            x = x.flatten().tolist()
        if not isinstance(x, list):
             x = [x]
        
        encrypted_vec = self.ts.ckks_vector(self.ctx, x)
        return Ciphertext(encrypted_vec, backend="tenseal")

    def decrypt(self, ct: Ciphertext):
        if ct.backend != "tenseal":
            raise ValueError("TenSEALContext cannot decrypt non-tenseal ciphertexts")
        
        decrypted_list = ct.data.decrypt()
        # Return as numpy array for consistency
        # If it was a scalar, it comes back as a list [val]
        if len(decrypted_list) == 1:
            return decrypted_list[0]
        return np.array(decrypted_list)

    def add(self, ct1, ct2):
        # Handle scalar addition if needed, or CT+CT
        val1 = ct1.data if isinstance(ct1, Ciphertext) else ct1
        val2 = ct2.data if isinstance(ct2, Ciphertext) else ct2
        
        # TenSEAL supports python operators
        return Ciphertext(val1 + val2, backend="tenseal")
        
    def sub(self, ct1, ct2):
        val1 = ct1.data if isinstance(ct1, Ciphertext) else ct1
        val2 = ct2.data if isinstance(ct2, Ciphertext) else ct2
        return Ciphertext(val1 - val2, backend="tenseal")

    def mul(self, ct1, ct2):
        val1 = ct1.data if isinstance(ct1, Ciphertext) else ct1
        val2 = ct2.data if isinstance(ct2, Ciphertext) else ct2
        return Ciphertext(val1 * val2, backend="tenseal")
        
    def scalar_mul(self, a, ct):
        return Ciphertext(ct.data * a, backend="tenseal")

    def l2_norm_sq(self, ct_a, ct_b):
        # ||x - y||^2 = sum((x-y)^2)
        # CKKS vector subtraction is coordinate-wise
        diff = self.sub(ct_a, ct_b)
        # Square
        squared_diff = diff.data ** 2 # Native TenSEAL squaring
        # Sum elements. TenSEAL vectors have a .sum() method? 
        # Actually .sum() might need rotation keys (Galois keys). We generated them.
        sum_val = squared_diff.sum()
        return Ciphertext(sum_val, backend="tenseal")

    def gaussian_kernel(self, ct_a, ct_b, sigma):
        # This is hard in pure HE. We will use the hybrid approach:
        # Decrypt squared distance -> compute exp on plain -> Encrypt/Return
        # (Simulating Client/KeyServer interaction for non-linear op)
        
        dist_sq_ct = self.l2_norm_sq(ct_a, ct_b)
        dist_sq = self.decrypt(dist_sq_ct) # Interaction step
        
        val = np.exp(-dist_sq / (2 * sigma**2))
        return self.encrypt([val]) # Return as ciphertext

    def negative_half_power(self, ct):
        # Computes x^(-1/2) via decryption (hybrid assumption)
        val = self.decrypt(ct)
        
        # Safety
        if val <= 1e-9: val = 1e-9 # Prevent div by zero
        
        res = 1.0 / np.sqrt(val)
        return self.encrypt([res])

    def exp(self, ct):
        val = self.decrypt(ct)
        return self.encrypt([np.exp(val)])

# Default to Mock for now, app.py can switch
def get_he_context(type="mock"):
    if type == "mock":
        return MockHEContext()
    elif type == "tenseal":
        return TenSEALContext()
    else:
        raise ValueError("Unknown HE type")
