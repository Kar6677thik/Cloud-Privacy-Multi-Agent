# Project Report: Cloud-Assisted Privacy-Preserving Spectral Clustering

## 1. Project Overview
This project is a **Privacy-Preserving Machine Learning (PPML)** prototype designed to demonstrate how sensitive data from multiple users can be aggregated and analyzed by a third-party cloud server **without revealing the raw data**.

It implements a **Spectral Clustering algorithm** that operates on encrypted data using a simulated **Homomorphic Encryption (HE)** scheme. This allows the cloud to compute clusters (grouping similar data points) while seeing only unintelligible ciphertext.

**Core Inspiration**: *“Cloud-Assisted Privacy-Preserving Spectral Clustering Algorithm Within a Multi-User Setting” (IEEE Access 2024)*.

---

## 2. Why is this Useful?
In the modern era, data is often distributed across multiple owners (e.g., hospitals with patient records, banks with financial data).
- **The Challenge**: These owners want to collaborate to train models or find patterns (clustering) but cannot share raw data due to privacy laws (GDPR, HIPAA) or competitive secrecy.
- **The Solution**: This system allows them to offload the computation to a powerful cloud server in an encrypted form. The cloud does the math, but learns nothing about the individual records.

---

## 3. System Architecture
The system consists of four logical entities:

### 1. Key Generation Center (KGC)
- **Role**: Trusted Authority.
- **Action**: Generates the Public Key (pk) for encryption and the Secret Key (sk) for decryption. Distributes keys to authorized parties.

### 2. Data Owners (DO)
- **Role**: Users who own private datasets.
- **Action**: 
  1. Clean and prepare their local data (CSV format).
  2. Encrypt the data using the Public Key.
  3. Upload the *Ciphertexts* to the Cloud Server.

### 3. Ciphertext Server (CS1)
- **Role**: The main cloud computation node.
- **Action**:
  1. Receives encrypted datasets from all Data Owners.
  2. Aggregates them into a single encrypted pool.
  3. perform homomorphic operations to compute the **Affinity Matrix** and **Laplacian Matrix**.
  4. Computes clustering results (Spectral Embedding).

### 4. Key Server (CS2) / Decryption Service
- **Role**: Holds the Secret Key.
- **Action**: In a full protocol, this server assists CS1 by performing partial decryptions required for intermediate steps (like comparing distances) without revealing the original data values to CS1. In this prototype, this is simulated alongside CS1.

---

## 4. Operational Workflow
The entire process follows these sequential steps:

1. **Initialization**: KGC generates keys.
2. **Data Ingestion**: System reads `user1.csv`, `user2.csv`, etc.
3. **Encryption**: Every number in the uploaded CSVs is wrapped in a `Ciphertext` object.
    - *Note*: In this prototype, we use a "Mock HE" layer that simulates the constraints (noise, limited operations) of real CKKS encryption without the heavy computational overhead.
4. **Affinity Calculation (Encrypted)**:
    - The server calculates the distance between every pair of points using the Gaussian Kernel formula: 
      $$W_{ij} = e^{\frac{-||x_i - x_j||^2}{2\sigma^2}}$$
    - This is done using homomorphic subtraction and multiplication.
5. **Laplacian Construction**:
    - The server computes the Normalized Laplacian matrix: $L = D^{-1/2} W D^{-1/2}$.
6. **Eigendecomposition**:
    - The server extracts the eigenvectors corresponding to the smallest eigenvalues of the Laplacian (or largest of the Affinity matrix).
    - This maps the data into a lower-dimensional "Spectral Embedding" space where clusters are easily separable.
7. **Clustering**:
    - Standard K-Means is run on the spectral embedding to assign final labels.

---

## 5. Input & Output

### Input
- **Files**: Multiple CSV files. Each file represents one Data Owner.
- **Content**: Numerical data (columns = features, rows = samples).
- **Parameters**: 
  - `k`: Number of clusters expected.
  - `sigma`: The bandwidth of the Gaussian kernel (controls how "local" the similarity is).

### Output
1. **Cluster Labels**: Which cluster each data point belongs to.
2. **Visualizations**:
   - **Cluster Plot**: 2D Scatter plot showing the grouped data.
   - **Spectral Embedding**: The data projected into the eigenvector space (often revealing the structure more clearly than the raw data).
   - **Matrices**: Heatmaps of the Affinity Matrix ($W$) and Laplacian ($L$) to inspect the mathematical structure.

---

## 6. Project Structure
- `he_layer.py`: **The "Safe" Box**. Defines the `Ciphertext` class and allowed operations (`add`, `mul`, etc.). It ensures we don't accidentally "peek" at the data in strict mode.
- `spectral_core.py`: **The Mathematician**. Contains the logic for building graphs from data and finding eigenvectors. It speaks the language of `he_layer`.
- `system_sim.py`: **The Conductor**. Connects the Data Owners, Server, and KGC. Orchestrates the flow.
- `app.py`: **The Face**. A Streamlit-based web dashboard to make the system easy to demonstrate and interact with.

## 7. Future Scope
This prototype serves as a feasibility study. For a production-grade deployment, the following changes would be made:
- Replace `he_layer.py` with a binding to a real library like **Microsoft SEAL** or **TenSEAL**.
- Distribute the components (CS1, CS2, DO) onto separate physical machines communicating via REST APIs.
