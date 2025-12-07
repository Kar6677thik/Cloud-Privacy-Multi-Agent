# Privacy-Preserving Spectral Clustering Prototype

##  Project Overview
Sytem Prototype for **B.Tech Final Year Capstone**.
This project demonstrates a **cloud-assisted privacy-preserving spectral clustering** workflow. It simulates a scenario where multiple data owners outsource their private data to a cloud server for clustering without revealing the raw data, using (mock) Homomorphic Encryption.

**Reference Paper**: *“Cloud-Assisted Privacy-Preserving Spectral Clustering Algorithm Within a Multi-User Setting” (IEEE Access 2024)*

##  Components
1. **Key Generation Center (KGC)**: Manages encryption keys.
2. **Data Owners (DO)**: Encrypt and upload private datasets.
3. **Ciphertext Server (CS1)**: Stores encrypted data and performs clustering computation.
4. **Key Server (CS2)**: Holds the secret key (simulated/required for specific decryption steps).

##  Security Model (Prototype)
- **Encryption**: Uses a mock implementation of **CKKS**-style Homomorphic Encryption (`he_layer.py`) to demonstrate the architectural flow.
- **Privacy**: The cloud server operates on `Ciphertext` objects. In a full production version, these would be mathematically secure polynomials. Here, they are wrapped arrays with tracked noise budgets.

##  How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app.py
```

### 3. Usage
- Open the Streamlit URL (usually `http://localhost:8501`).
- Upload multiple CSV files (or generate synthetic data via sidebar).
- Adjust parameters `k` (clusters) and `sigma` (affinity).
- Click **"Run Clustering"**.
- View visualizations of the clusters, spectral embedding, and internal matrices.

##  File Structure
- `he_layer.py`: Mock HE implementation.
- `spectral_core.py`: Logic for Spectral Clustering (Affinity, Laplacian, Eigendecomposition).
- `system_sim.py`: Classes representing the multi-user system architecture.
- `app.py`: Streamlit User Interface.
