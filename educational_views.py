import streamlit as st
import numpy as np
import pandas as pd
import time
from he_layer import get_he_context, Ciphertext
import plotly.graph_objects as go
import plotly.express as px

def render_encryption_deep_dive(he_backend="mock"):
    """
    Renders the interactive educational deep dive into the encryption process.
    """
    st.title("Encryption Deep Dive")
    st.markdown("""
    Welcome to the **Protocol Internals**. Here you can interactively explore how 
    Homomorphic Encryption (CKKS) protects data while allowing computation.
    """)
    
    # --- Step 1: Key Generation ---
    st.header("1. Key Generation & Setup")
    
    col_keys_1, col_keys_2 = st.columns([2, 1])
    
    with col_keys_1:
        st.markdown("""
        In a secure protocol, we first need a cryptographic context.
        - **Public Key (PK)**: Used by Data Owners to encrypt data.
        - **Secret Key (SK)**: Kept safe by the Key Server (KS) to decrypt results.
        - **Relinearization Keys (RLK)**: Used during multiplication to keep ciphertext size manageable.
        """)
    
    with col_keys_2:
        if st.button("Generate Keys", key="edu_keygen"):
            with st.spinner("Generating High-Security Keys (CKKS)..."):
                time.sleep(1.0) # Artificial delay for effect
                st.session_state['edu_keys_generated'] = True
    
    if st.session_state.get('edu_keys_generated'):
        st.success("✅ Keys Generated Successfully!")
        
        # Visualize Key Structures (conceptual)
        k_col1, k_col2 = st.columns(2)
        with k_col1:
            st.info("**Public Key**\n\n`pk = (b, a)`\n\nAvailable to everyone.")
        with k_col2:
            st.error("**Secret Key**\n\n`sk = s`\n\n**TOP SECRET**. Never leaves the Key Server.")
            
        # Context Info
        with st.expander("Show Crypto Parameters"):
            st.code("""
Scheme: CKKS (Homomorphic Encryption for Arithmetic of Approximate Numbers)
Poly Modulus Degree: 8192
Coeff Modulus Sizes: [60, 40, 40, 60] bits
Security Level: ~128-bit
            """)
            
    # Need keys to proceed
    if not st.session_state.get('edu_keys_generated'):
        st.warning("Please generate keys to proceed to Encryption.")
        return

    st.markdown("---")

    # --- Step 2: Encryption Visualization ---
    st.header("2. Encryption Process")
    
    st.markdown("Let's encrypt a single number. Watch how a simple value turns into complex polynomial noise.")
    
    col_enc_in, col_enc_viz = st.columns(2)
    
    with col_enc_in:
        user_input_val = st.number_input("Enter a value (x)", value=3.14159, format="%.5f")
        if st.button("Encrypt x"):
            ctx = get_he_context(he_backend)
            
            # Setup context if needed (for mock it's stateless mostly, for tenseal handled internally)
            if he_backend == "tenseal":
                ctx.keygen() # Re-init for this demo scope
                
            # Encrypt
            st.session_state['edu_ctx'] = ctx
            st.session_state['edu_ct_x'] = ctx.encrypt([user_input_val])
            st.session_state['edu_val_x'] = user_input_val
            
    if 'edu_ct_x' in st.session_state:
        ct = st.session_state['edu_ct_x']
        
        with col_enc_viz:
            st.markdown("### Ciphertext Representation")
            if he_backend == "mock":
                st.info("In **MOCK** mode, we just wrap the data. In real HE, this would be unrecognizable.")
                st.code(f"Wrapped Value: {ct.data}")
            else:
                st.markdown("In **TenSEAL (Real HE)**, the data is a vector of polynomials.")
                # Show some "gibberish" or hex representation to look cool
                st.code(f"Ciphertext Object:\n{str(ct.data)[:100]}...\n[Binary Blob of ~100KB]")
        
        # Visualization of "Noise"
        st.markdown("#### Message + Noise")
        st.markdown("Encryption effectively hides your number $x$ in a high-dimensional space.")
        
        # Interactive Graph: Signal vs Noise (Conceptual)
        x_vals = np.linspace(0, 10, 100)
        clean_sig = np.sin(x_vals) # Just art
        noisy_sig = clean_sig + np.random.normal(0, 0.5, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=noisy_sig, mode='lines', name='Ciphertext (Look random)', line=dict(color='red', width=1)))
        fig.add_trace(go.Scatter(y=clean_sig, mode='lines', name='Message (Hidden)', line=dict(color='green', width=3, dash='dash')))
        fig.update_layout(title="Conceptual Visualization: Message Hidden in Noise", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # --- Step 3: Homomorphic Computation ---
    if 'edu_ct_x' in st.session_state:
        st.markdown("---")
        st.header("3. Homomorphic Computation")
        st.markdown("""
        **The Magic:** We can do math on the *Ciphertext* without ever decrypting it. 
        The Cloud Server sees only the "Red noisy line" above, but makes changes that affect the "Green hidden line".
        """)
        
        col_op_1, col_op_2 = st.columns(2)
        
        with col_op_1:
            op_type = st.radio("Operation", ["Add Scalar", "Multiply Scalar", "Square (x^2)"])
            operand = 0
            if "Scalar" in op_type:
                operand = st.number_input("Scalar Value (y)", value=2.0)
                
            if st.button("Compute on Ciphertext"):
                ctx = st.session_state['edu_ctx']
                ct_res = None
                
                if op_type == "Add Scalar":
                    ct_res = ctx.add(st.session_state['edu_ct_x'], operand)
                    formula = f"Enc(x) + {operand}"
                    expected = st.session_state['edu_val_x'] + operand
                elif op_type == "Multiply Scalar":
                    ct_res = ctx.scalar_mul(operand, st.session_state['edu_ct_x'])
                    formula = f"Enc(x) * {operand}"
                    expected = st.session_state['edu_val_x'] * operand
                elif op_type == "Square (x^2)":
                    ct_res = ctx.mul(st.session_state['edu_ct_x'], st.session_state['edu_ct_x'])
                    formula = "Enc(x) * Enc(x)"
                    expected = st.session_state['edu_val_x'] ** 2
                    
                st.session_state['edu_ct_res'] = ct_res
                st.session_state['edu_formula'] = formula
                st.session_state['edu_expected'] = expected

        with col_op_2:
            if 'edu_ct_res' in st.session_state:
                st.success(f"Computed: `{st.session_state['edu_formula']}`")
                
                st.markdown("### Resulting Ciphertext")
                st.markdown("It still looks like random noise/encrypted data.")
                
                # Visualizing "Noise Growth" (Conceptual)
                st.metric("Noise Budget Consumption", "Low" if "Add" in st.session_state.get('edu_formula', '') else "Medium")
                st.progress(10 if "Add" in st.session_state.get('edu_formula', '') else 40)
                st.caption("Multiplications consume more noise budget than additions.")

    # --- Step 4: Decryption ---
    if 'edu_ct_res' in st.session_state:
        st.markdown("---")
        st.header("4. Decryption & Verification")
        
        st.markdown("Finally, the Key Server uses the Secret Key (SK) to recover the result.")
        
        if st.button("Decrypt Result"):
            ctx = st.session_state['edu_ctx']
            decrypted_val = ctx.decrypt(st.session_state['edu_ct_res'])
            
            # Unpack if list
            if isinstance(decrypted_val, (list, np.ndarray)):
                if len(decrypted_val) > 0:
                    decrypted_val = decrypted_val[0]
            
            st.session_state['edu_decrypted'] = decrypted_val
            
    if 'edu_decrypted' in st.session_state:
        d_val = st.session_state['edu_decrypted']
        e_val = st.session_state['edu_expected']
        
        # Display with high precision
        col_res_1, col_res_2 = st.columns(2)
        
        with col_res_1:
            st.metric("Decrypted Value", f"{d_val:.8f}")
        with col_res_2:
            st.metric("Expected (Plaintext)", f"{e_val:.8f}")
            
        # Error Analysis
        err = abs(d_val - e_val)
        st.info(f"Approximation Error: `{err:.2e}`")
        if err < 1e-3:
            st.success("✅ Success! The math worked on encrypted data.")
        else:
            st.warning("⚠️ High error. CKKS is approximate, but this seems high.")
