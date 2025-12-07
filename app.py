import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from system_sim import simulate_multi_user_system

# Page Config
st.set_page_config(
    page_title="Privacy-Preserving Spectral Clustering",
    page_icon="🔒",
    layout="wide"
)

# Title and Intro
st.title("🔒 Prototype Privacy-Preserving Spectral Clustering System")
st.markdown("""
**B.Tech Final Year Capstone Project**  
*Inspired by: Cloud-Assisted Privacy-Preserving Spectral Clustering Algorithm Within a Multi-User Setting (IEEE Access 2024)*

---
This dashboard demonstrates a secure cloud-assisted clustering workflow where:
1. **Data Owners** upload encrypted private data.
2. **Cloud Server** computes clustering on the aggregated encrypted pool.
3. **Privacy** is preserved via (simulated) Homomorphic Encryption.
""")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")

# Example Datasets Generation
if st.sidebar.button("Generate Synthetic Data"):
    # Create simple concentric circles or blobs
    from sklearn.datasets import make_blobs
    data1, _ = make_blobs(n_samples=50, centers=2, n_features=2, random_state=1)
    data2, _ = make_blobs(n_samples=50, centers=2, n_features=2, random_state=2)
    # Shift data2
    data2 += 5
    
    st.session_state['uploaded_files'] = {
        "Owner_1_Synthetic.csv": data1,
        "Owner_2_Synthetic.csv": data2
    }
    st.sidebar.success("Generated 2 synthetic datasets!")

k_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=3)
sigma = st.sidebar.slider("Gaussian Sigma (σ)", min_value=0.1, max_value=5.0, value=1.0)

# File Upload
st.header("1. Data Upload (Data Owners)")
uploaded_files = st.file_uploader("Upload CSV files (one per user)", accept_multiple_files=True)

data_owners_data = {}

# Priority to session state (synthetic) if files not uploaded
if not uploaded_files and 'uploaded_files' in st.session_state:
    data_owners_data = st.session_state['uploaded_files']
    st.info("Using synthetic data. Upload files to override.")
    
elif uploaded_files:
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            # Assume numerical data only. Drop non-numeric.
            df_numeric = df.select_dtypes(include=[np.number])
            if df_numeric.empty:
                st.error(f"File {f.name} has no numeric columns.")
                continue
            data_owners_data[f.name] = df_numeric.values
        except Exception as e:
            st.error(f"Error reading {f.name}: {e}")

# Display Input Data
if data_owners_data:
    cols = st.columns(len(data_owners_data))
    all_data_list = []
    
    for idx, (name, data) in enumerate(data_owners_data.items()):
        all_data_list.append(data)
        with cols[idx % len(cols)]:
            st.subheader(f"{name}")
            st.write(f"Shape: {data.shape}")
            st.dataframe(pd.DataFrame(data).head(5))
            
    # Run Simulation
    st.header("2. Cloud Computation (Mock HE)")
    if st.button("🚀 Encrypt, Upload & Run Clustering"):
        with st.spinner("Encrypting data, computing Affinity Matrix & Laplacian..."):
            
            try:
                results = simulate_multi_user_system(data_owners_data, k=k_clusters, sigma=sigma)
                
                labels = results['labels']
                Y = results['Y']
                W = results['W']
                L = results['L']
                owner_labels = results['owner_labels']
                
                aggregated_data = np.vstack(all_data_list)
                
                st.success("Computation Complete!")
                
                # Visualizations
                st.header("3. Results Analysis")
                
                tab1, tab2, tab3 = st.tabs(["Clustering Result", "Spectral Embedding", "Internal Matrices"])
                
                with tab1:
                    st.subheader("Final Clusters on Aggregated Data")
                    fig, ax = plt.subplots()
                    if aggregated_data.shape[1] >= 2:
                        scatter = ax.scatter(aggregated_data[:, 0], aggregated_data[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k')
                        plt.colorbar(scatter, label='Cluster Label')
                        ax.set_title("Clustering Output (Plaintext View)")
                        ax.set_xlabel("Feature 1")
                        ax.set_ylabel("Feature 2")
                    else:
                        ax.text(0.5, 0.5, "Data < 2D, cannot plot scatter", ha='center')
                    st.pyplot(fig)
                    
                with tab2:
                    st.subheader("Spectral Embedding (First 2 Eigenvectors)")
                    # Y has k columns. We plot first 2.
                    fig2, ax2 = plt.subplots()
                    if Y.shape[1] >= 2:
                        # Color by ground truth owner to see if structure is preserved or just by output cluster
                        scatter2 = ax2.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', alpha=0.7)
                        ax2.set_xlabel("Eigenvector 1")
                        ax2.set_ylabel("Eigenvector 2")
                        ax2.set_title("Projected Data in Spectral Domain")
                    else:
                        ax2.text(0.5, 0.5, "k < 2, embedding 1D", ha='center')
                    st.pyplot(fig2)
                    
                with tab3:
                    col_w, col_l = st.columns(2)
                    with col_w:
                        st.subheader("Affinity Matrix W")
                        st.image(W / np.max(W), clamp=True, caption=f"Shape: {W.shape}", output_format="PNG") # Simple normalization for viz
                        # Or use heatmap
                        fig_w, ax_w = plt.subplots()
                        im_w = ax_w.imshow(W, cmap='hot', interpolation='nearest')
                        plt.colorbar(im_w)
                        ax_w.set_title("W (Affinity)")
                        st.pyplot(fig_w)
                        
                    with col_l:
                        st.subheader("Normalized Laplacian L")
                        fig_l, ax_l = plt.subplots()
                        im_l = ax_l.imshow(L, cmap='coolwarm', interpolation='nearest')
                        plt.colorbar(im_l)
                        ax_l.set_title("L (Laplacian)")
                        st.pyplot(fig_l)
                        
            except Exception as e:
                st.error(f"An error occurred during clustering: {e}")
                # Print stack trace for debugging if needed
                import traceback
                st.text(traceback.format_exc())

else:
    st.info("👋 Upload data files or click 'Generate Synthetic Data' to start.")
