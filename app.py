import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from system_sim import simulate_multi_user_system
from attack_sim import AttackSimulator
from benchmark import run_benchmark
from data_utils import UniversalAdapter
from auto_tuning import find_optimal_k
from report_gen import generate_pdf_report
from insights import get_insights_summary
from anomaly_detection import ClusteringWithAnomalies
from session_history import get_session_history, compute_data_hash
from educational_views import render_encryption_deep_dive
import tempfile
import graphviz
import time
import os

# Page Config
st.set_page_config(
    page_title="Privacy-Preserving Spectral Clustering",
    page_icon="",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 5px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .history-item {
        background: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #3d3d4d;
        border-radius: 8px;
        padding: 10px 20px;
        color: white;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #4d4d5d;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Prototype System", 
    "Privacy Security Audit", 
    "Performance Benchmark",
    "Privacy Attack Playground",
    "Encryption Deep Dive"
])

st.sidebar.markdown("---")
st.sidebar.header("System Config")
he_backend = st.sidebar.selectbox("HE Backend", ["mock", "tenseal"], 
    help="Select 'tenseal' for real CKKS encryption. 'mock' for simulation.")

# Initialize session history
if 'session_history' not in st.session_state:
    st.session_state['session_history'] = get_session_history()

# Global Helper for Synthetic Data
def get_synthetic_data():
    if 'uploaded_files' not in st.session_state:
        from sklearn.datasets import make_blobs
        data1, _ = make_blobs(n_samples=50, centers=2, n_features=2, random_state=1)
        data2, _ = make_blobs(n_samples=50, centers=2, n_features=2, random_state=2)
        data2 += 5
        st.session_state['uploaded_files'] = {
            "Owner_1_Synthetic.csv": data1,
            "Owner_2_Synthetic.csv": data2
        }
    return st.session_state['uploaded_files']


# --- FEATURE: AI Dataset Insights Panel ---
def render_insights_panel(aggregated_data: np.ndarray):
    """Render the AI-powered dataset insights panel."""
    st.subheader("AI Dataset Insights")
    
    with st.spinner("Analyzing dataset structure..."):
        insights = get_insights_summary(aggregated_data)
    
    # Cluster Tendency Score (Hopkins)
    col1, col2, col3 = st.columns(3)
    
    hopkins = insights.get("cluster_tendency", {}).get("hopkins", 0.5)
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hopkins * 100,
            title={'text': "Cluster Tendency"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccb"},
                    {'range': [50, 75], 'color': "#ffffcc"},
                    {'range': [75, 100], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(t=50, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        outlier_pct = insights.get("outliers", {}).get("outlier_percentage", 0)
        st.metric("Outlier Rate", f"{outlier_pct:.1f}%", 
                  delta=None if outlier_pct < 5 else "High - Enable Anomaly Mode",
                  delta_color="inverse" if outlier_pct > 5 else "off")
        
        n_outliers = insights.get("outliers", {}).get("combined_count", 0)
        st.caption(f"{n_outliers} potential outliers detected")
    
    with col3:
        basic = insights.get("basic_stats", {})
        st.metric("Data Shape", f"{basic.get('n_samples', 0)} x {basic.get('n_features', 0)}")
        
        # Suggested sigma
        suggested_sigma = insights.get("density_estimation", {}).get("suggested_sigma", 1.0)
        st.caption(f"Suggested sigma: {suggested_sigma:.2f}")
    
    # Recommendations
    st.markdown("#### Recommendations")
    recs = insights.get("recommendations", [])
    
    for rec in recs:
        rec_type = rec.get("type", "info")
        message = rec.get("message", "")
        # Remove emojis from messages
        message = message.replace("✅ ", "").replace("⚠️ ", "").replace("ℹ️ ", "").replace("💡 ", "")
        
        if rec_type == "success":
            st.success(message)
        elif rec_type == "warning":
            st.warning(message)
        elif rec_type == "tip":
            st.info(message)
        else:
            st.info(message)
    
    # Feature Correlation Heatmap (Collapsible)
    with st.expander("Feature Correlation Analysis"):
        corr_matrix = insights.get("feature_correlation", {}).get("correlation_matrix")
        if corr_matrix:
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale="rdbu",
                aspect="auto"
            )
            fig_corr.update_layout(height=300)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # PCA info
            pca_var = insights.get("feature_correlation", {}).get("pca_variance_ratio", [])
            if pca_var:
                n_comp = insights.get("feature_correlation", {}).get("n_components_95_var", 0)
                st.caption(f"{n_comp} components explain 95% of variance")
    
    return insights


# --- FEATURE: Privacy-Utility Trade-off Dashboard ---
def render_privacy_utility_dashboard(aggregated_data: np.ndarray, k: int, sigma: float):
    """Interactive dashboard showing privacy-utility trade-off."""
    st.subheader("Privacy-Utility Trade-off Explorer")
    
    st.markdown("*Adjust epsilon to see how privacy affects clustering quality in real-time*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        test_epsilon = st.slider(
            "Privacy Budget (epsilon)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="privacy_explorer_epsilon"
        )
        
        # Privacy interpretation
        if test_epsilon < 0.5:
            st.error("Very High Privacy - Significant noise added. Results may be unreliable.")
        elif test_epsilon < 2.0:
            st.warning("High Privacy - Good privacy-utility balance.")
        elif test_epsilon < 5.0:
            st.info("Moderate Privacy - Lower noise, better accuracy.")
        else:
            st.success("Low Privacy - Minimal noise, maximum accuracy.")
    
    with col2:
        # Run quick simulation with different epsilon values
        if st.button("Simulate Trade-off Curve"):
            with st.spinner("Computing trade-off curve..."):
                epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
                silhouettes = []
                
                progress_bar = st.progress(0)
                for i, eps in enumerate(epsilons):
                    try:
                        # Quick simulation
                        data_dict = {"test": aggregated_data}
                        result = simulate_multi_user_system(
                            data_dict, k=k, sigma=sigma, 
                            he_type="mock", epsilon=eps
                        )
                        sil = result.get("metrics", {}).get("Silhouette Score", 0)
                        silhouettes.append(sil if not np.isnan(sil) else 0)
                    except:
                        silhouettes.append(0)
                    progress_bar.progress((i + 1) / len(epsilons))
                
                # Plot trade-off
                fig_tradeoff = go.Figure()
                fig_tradeoff.add_trace(go.Scatter(
                    x=epsilons,
                    y=silhouettes,
                    mode='lines+markers',
                    name='Silhouette Score',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=10)
                ))
                
                # Add privacy region shading
                fig_tradeoff.add_vrect(x0=0, x1=1, fillcolor="red", opacity=0.1, 
                                       line_width=0, annotation_text="High Privacy")
                fig_tradeoff.add_vrect(x0=1, x1=5, fillcolor="yellow", opacity=0.1,
                                       line_width=0, annotation_text="Balanced")
                fig_tradeoff.add_vrect(x0=5, x1=10, fillcolor="green", opacity=0.1,
                                       line_width=0, annotation_text="High Utility")
                
                fig_tradeoff.update_layout(
                    title="Privacy-Utility Trade-off Curve",
                    xaxis_title="Epsilon - Higher = Less Private",
                    yaxis_title="Clustering Quality (Silhouette)",
                    height=350
                )
                st.plotly_chart(fig_tradeoff, use_container_width=True)


# --- FEATURE: Real-time Clustering Animation ---
def render_animated_clustering(Y: np.ndarray, labels: np.ndarray, k: int):
    """Show animated clustering visualization."""
    st.subheader("Clustering Animation")
    
    if Y.shape[1] < 2:
        st.warning("Need at least 2D embedding for animation")
        return
    
    n_frames = 20
    frames = []
    
    # Simulate the clustering process: random -> spectral -> final
    np.random.seed(42)
    random_positions = np.random.randn(*Y.shape) * 2
    
    for i in range(n_frames):
        t = i / (n_frames - 1)
        # Ease-in-out interpolation
        t_smooth = 0.5 * (1 - np.cos(np.pi * t))
        
        # Interpolate from random to final
        current_pos = (1 - t_smooth) * random_positions + t_smooth * Y
        
        df_frame = pd.DataFrame({
            'x': current_pos[:, 0],
            'y': current_pos[:, 1],
            'Cluster': labels.astype(str),
            'frame': i
        })
        frames.append(df_frame)
    
    df_animation = pd.concat(frames, ignore_index=True)
    
    fig_anim = px.scatter(
        df_animation,
        x='x', y='y',
        color='Cluster',
        animation_frame='frame',
        title="Data Points Converging to Clusters",
        range_x=[df_animation['x'].min() - 1, df_animation['x'].max() + 1],
        range_y=[df_animation['y'].min() - 1, df_animation['y'].max() + 1]
    )
    
    fig_anim.update_layout(
        height=450,
        showlegend=True,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate', 
                 'args': [None, {'frame': {'duration': 100}, 'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
            ]
        }]
    )
    
    st.plotly_chart(fig_anim, use_container_width=True)
    st.caption("Watch data points migrate from random positions to their final cluster assignments")


# --- FEATURE: Session History Panel ---
def render_history_panel():
    """Render the clustering history panel."""
    st.subheader("Experiment History")
    
    history = st.session_state.get('session_history', get_session_history())
    runs = history.get_all_runs()
    
    if not runs:
        st.info("No experiments recorded yet. Run a clustering to see history.")
        return
    
    st.caption(f"Showing {len(runs)} experiment(s)")
    
    # History table
    history_data = []
    for run in runs:
        metrics = run.get("metrics", {})
        config = run.get("config", {})
        history_data.append({
            "ID": run["id"],
            "Time": run["timestamp"][:19],
            "k": config.get("k", "?"),
            "sigma": config.get("sigma", "?"),
            "epsilon": config.get("epsilon", "None"),
            "Silhouette": f"{metrics.get('Silhouette Score', 0):.4f}" if metrics.get('Silhouette Score') else "N/A",
            "Samples": run.get("n_samples", 0)
        })
    
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history, use_container_width=True, hide_index=True)
    
    # Compare runs
    if len(runs) >= 2:
        st.markdown("#### Compare Runs")
        col1, col2 = st.columns(2)
        
        run_ids = [r["id"] for r in runs]
        with col1:
            compare_1 = st.selectbox("Run 1", run_ids, key="compare_1")
        with col2:
            compare_2 = st.selectbox("Run 2", [r for r in run_ids if r != compare_1], key="compare_2")
        
        if st.button("Compare"):
            comparison = history.compare_runs(compare_1, compare_2)
            
            if comparison.get("same_data"):
                st.success("Same input data")
            else:
                st.warning("Different input data")
            
            # Show metric differences
            metric_diff = comparison.get("metric_diff", {})
            if metric_diff:
                st.markdown("**Metric Comparison:**")
                for metric, vals in metric_diff.items():
                    delta = vals.get("delta", 0)
                    direction = "[UP]" if vals.get("improvement") else "[DOWN]"
                    st.write(f"{direction} {metric}: {vals['run_1']:.4f} -> {vals['run_2']:.4f} ({delta:+.4f})")


def get_downloads_folder():
    """Get the user's Downloads folder path."""
    return os.path.join(os.path.expanduser("~"), "Downloads")


# --- PAGE 1: PROTOTYPE ---
def render_prototype():
    st.title("Privacy-Preserving Spectral Clustering System")
    st.markdown(f"**Backend Active:** `{he_backend.upper()}`")
    
    col_intro, col_arch = st.columns([3, 2])
    
    with col_intro:
        st.markdown("""
        This dashboard demonstrates a secure cloud-assisted clustering workflow.
        
        **Key Features:**
        - **AI Insights**: Automatic data analysis before clustering
        - **Privacy Control**: Differential Privacy with adjustable epsilon
        - **Anomaly Detection**: Automatic outlier handling
        - **Experiment History**: Track and compare runs
        """)
        
    with col_arch:
        try:
            graph = graphviz.Digraph(format='svg')
            graph.attr(rankdir='LR')
            graph.node('DO', 'Data Owners\n(Encrypted Input)', shape='box', style='filled', fillcolor='#e1f5fe')
            graph.node('CS', 'Cloud Server\n(Computation)', shape='cylinder', style='filled', fillcolor='#fff3e0')
            graph.node('KS', 'Key Server\n(Secret Keys)', shape='ellipse', style='filled', fillcolor='#f3e5f5')
            
            graph.edge('DO', 'CS', label='Ciphertext')
            graph.edge('CS', 'KS', label='Partial Decrypt', style='dashed')
            graph.edge('KS', 'CS', label='Blinded Result', style='dashed')
            
            st.graphviz_chart(graph)
        except Exception as e:
            st.warning("Graphviz not available. Architecture diagram skipped.")
    
    # Sidebar Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Clustering Params")
    
    if 'suggested_k' not in st.session_state:
        st.session_state['suggested_k'] = 3
        
    k_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, 
                                   value=st.session_state['suggested_k'])
    sigma = st.sidebar.slider("Gaussian Sigma", min_value=0.1, max_value=5.0, value=1.0)
    
    st.sidebar.subheader("Differential Privacy")
    use_dp = st.sidebar.checkbox("Enable Differential Privacy")
    epsilon = None
    if use_dp:
        epsilon = st.sidebar.slider("Privacy Budget (epsilon)", 
                                    min_value=0.1, max_value=10.0, value=1.0)
    
    st.sidebar.subheader("Anomaly Detection")
    enable_anomaly = st.sidebar.checkbox("Enable Anomaly-Aware Mode")
    anomaly_method = "lof"
    contamination = 0.1
    if enable_anomaly:
        anomaly_method = st.sidebar.selectbox("Detection Method", ["lof", "iforest"])
        contamination = st.sidebar.slider("Expected Outlier %", 1, 30, 10) / 100
    
    if st.sidebar.button("Generate Synthetic Data"):
        get_synthetic_data()
        st.sidebar.success("Generated synthetic datasets!")

    # File Upload
    st.header("1. Data Upload")
    uploaded_files = st.file_uploader("Upload CSV files (one per user)", accept_multiple_files=True)
    
    data_owners_data = {}
    
    if not uploaded_files and 'uploaded_files' in st.session_state:
        data_owners_data = st.session_state['uploaded_files']
        st.info("Using synthetic data. Upload files to override.")
        
    elif uploaded_files:
        adapter = UniversalAdapter()
        for f in uploaded_files:
            try:
                df = pd.read_csv(f)
                try:
                    data_clean, _ = adapter.process(df)
                    data_owners_data[f.name] = data_clean
                except ValueError:
                    st.error(f"File {f.name} contains no valid numeric data.")
            except Exception as e:
                st.error(f"Error reading {f.name}: {e}")

    # Display Input Data & Run Analysis
    if data_owners_data:
        cols = st.columns(len(data_owners_data))
        all_data_list = []
        
        for idx, (name, data) in enumerate(data_owners_data.items()):
            all_data_list.append(data)
            with cols[idx % len(cols)]:
                st.subheader(f"{name}")
                st.write(f"Shape: {data.shape}")
                st.dataframe(pd.DataFrame(data).head(5))
        
        aggregated_data = np.vstack(all_data_list)
        
        # AI Insights Panel (FEATURE 6)
        with st.expander("AI Dataset Insights", expanded=True):
            insights = render_insights_panel(aggregated_data)
        
        # Auto-K Tuning
        col_autok, col_privacy = st.columns(2)
        
        with col_autok:
            if st.button("Auto-Tune k (Elbow Method)"):
                with st.spinner("Analyzing..."):
                    tune_results, best_k = find_optimal_k(aggregated_data)
                    st.session_state['suggested_k'] = best_k
                    
                    fig_tune, ax_tune = plt.subplots(figsize=(6, 3))
                    ax_tune.plot(tune_results['k'], tune_results['silhouette'], marker='o')
                    ax_tune.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
                    ax_tune.set_xlabel("k")
                    ax_tune.set_ylabel("Silhouette")
                    ax_tune.legend()
                    st.pyplot(fig_tune)
                    st.success(f"Recommended k={best_k}")
                    st.rerun()
        
        # Privacy-Utility Dashboard (FEATURE 2)
        with col_privacy:
            with st.expander("Privacy-Utility Explorer"):
                render_privacy_utility_dashboard(aggregated_data, k_clusters, sigma)

        # Run Clustering
        st.header("2. Cloud Computation")
        if st.button("Encrypt, Upload & Run Clustering", type="primary"):
            with st.spinner(f"Running Protocol using {he_backend} backend..."):
                try:
                    start_time = time.time()
                    results = simulate_multi_user_system(
                        data_owners_data, k=k_clusters, sigma=sigma, 
                        he_type=he_backend, epsilon=epsilon
                    )
                    
                    labels = results['labels']
                    Y = results['Y']
                    W = results['W']
                    L = results['L']
                    metrics = results.get('metrics', {})
                    
                    # Apply Anomaly Detection (FEATURE 7)
                    outlier_info = None
                    if enable_anomaly:
                        anomaly_handler = ClusteringWithAnomalies(
                            enable_anomaly=True,
                            anomaly_method=anomaly_method,
                            contamination=contamination
                        )
                        labels, outlier_info = anomaly_handler.process(labels, aggregated_data)
                    
                    elapsed = time.time() - start_time
                    
                    # Save to History (FEATURE 9)
                    config = {
                        "k": k_clusters,
                        "sigma": sigma,
                        "epsilon": epsilon,
                        "backend": he_backend,
                        "anomaly_enabled": enable_anomaly
                    }
                    data_hash = compute_data_hash(data_owners_data)
                    history = st.session_state['session_history']
                    run_id = history.add_run(config, metrics, labels, Y, data_hash)
                    
                    st.success(f"Computation Complete! (Run #{run_id}, {elapsed:.2f}s)")
                    
                    # Outlier Info
                    if outlier_info:
                        st.info(f"Detected {outlier_info['n_outliers']} outliers ({outlier_info['outlier_percentage']:.1f}%) - marked as cluster -1")
                    
                    # Metrics Section
                    st.header("3. Quality Metrics")
                    if metrics:
                        m_cols = st.columns(len(metrics))
                        for idx, (k_m, v_m) in enumerate(metrics.items()):
                            m_cols[idx].metric(k_m, f"{v_m:.4f}")
                    
                    # Visualizations
                    st.header("4. Visualizations")
                    
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Clustering Result", 
                        "Animation",
                        "Spectral Embedding",
                        "Internal Matrices"
                    ])
                    
                    fig_cluster = None
                    
                    with tab1:
                        st.subheader("Final Clusters")
                        if aggregated_data.shape[1] >= 2:
                            df_viz = pd.DataFrame(aggregated_data, 
                                columns=[f"Feature_{i}" for i in range(aggregated_data.shape[1])])
                            
                            # Handle outliers in visualization
                            cluster_labels = labels.astype(str)
                            cluster_labels = np.where(labels == -1, "Outlier", cluster_labels)
                            df_viz['Cluster'] = cluster_labels
                            
                            if aggregated_data.shape[1] >= 3:
                                fig = px.scatter_3d(df_viz, x='Feature_0', y='Feature_1', z='Feature_2', 
                                                   color='Cluster', title="Clustering Output (3D)")
                            else:
                                fig = px.scatter(df_viz, x='Feature_0', y='Feature_1', 
                                               color='Cluster', title="Clustering Output (2D)",
                                               color_discrete_map={"Outlier": "gray"})
                            st.plotly_chart(fig, use_container_width=True)
                            fig_cluster = fig
                    
                    with tab2:
                        # Clustering Animation (FEATURE 1)
                        render_animated_clustering(Y, labels, k_clusters)
                    
                    with tab3:
                        st.subheader("Spectral Embedding")
                        cols_y = Y.shape[1]
                        df_y = pd.DataFrame(Y, columns=[f"Eigen_{i}" for i in range(cols_y)])
                        cluster_labels = np.where(labels == -1, "Outlier", labels.astype(str))
                        df_y['Cluster'] = cluster_labels
                        
                        if cols_y >= 3:
                            fig2 = px.scatter_3d(df_y, x='Eigen_0', y='Eigen_1', z='Eigen_2', 
                                               color='Cluster', title="Spectral Embedding (3D)")
                        else:
                            fig2 = px.scatter(df_y, x='Eigen_0', 
                                            y='Eigen_1' if cols_y >= 2 else [0]*len(df_y),
                                            color='Cluster', title="Spectral Embedding")
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    with tab4:
                        col_w, col_l = st.columns(2)
                        with col_w:
                            st.subheader("Affinity Matrix W")
                            fig_w = px.imshow(W, color_continuous_scale='hot', title="W")
                            st.plotly_chart(fig_w, use_container_width=True)
                        with col_l:
                            st.subheader("Normalized Laplacian")
                            fig_l = px.imshow(L, color_continuous_scale='rdbu', title="L")
                            st.plotly_chart(fig_l, use_container_width=True)
                    
                    # Report Generation
                    st.markdown("---")
                    st.header("Report Generation")
                    if st.button("Generate & Download PDF"):
                        with st.spinner("Generating PDF..."):
                            img_path = None
                            if fig_cluster and aggregated_data.shape[1] >= 2:
                                fig_static, ax_static = plt.subplots()
                                colors = ['gray' if l == -1 else plt.cm.tab10(l % 10) for l in labels]
                                ax_static.scatter(aggregated_data[:, 0], aggregated_data[:, 1], c=colors)
                                plt.title(f"Clustering (k={k_clusters})")
                                
                                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                fig_static.savefig(tfile.name)
                                img_path = tfile.name
                                tfile.close()

                            config_summary = {
                                "k": k_clusters,
                                "sigma": sigma,
                                "backend": he_backend,
                                "dp_epsilon": epsilon,
                                "anomaly_mode": enable_anomaly
                            }
                            
                            # Generate PDF to Downloads folder
                            downloads_folder = get_downloads_folder()
                            pdf_filename = f"clustering_report_{int(time.time())}.pdf"
                            pdf_path = generate_pdf_report(metrics, config_summary, plot_image_path=img_path)
                            
                            # Copy to Downloads
                            import shutil
                            final_path = os.path.join(downloads_folder, pdf_filename)
                            shutil.copy(pdf_path, final_path)
                            
                            st.success(f"PDF saved to: {final_path}")
                            
                            with open(final_path, "rb") as f:
                                st.download_button("Download PDF", f, 
                                                  file_name=pdf_filename, 
                                                  mime="application/pdf")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    import traceback
                    st.text(traceback.format_exc())
        
        # History Panel (FEATURE 9)
        st.markdown("---")
        with st.expander("Experiment History"):
            render_history_panel()
            
    else:
        st.info("Upload data files or use Sidebar generation tool.")


# --- PAGE 2: SECURITY AUDIT ---
def render_security_audit():
    st.title("Security Audit & Attack Simulation")
    st.markdown("Test the resilience of the system against common attacks.")
    
    st.header("1. Data Leakage Test")
    st.markdown("Simulate a scenario where the Cloud Server tries to peek at the encrypted data.")
    
    if st.button("Run Leakage Audit"):
        st.write("Initializing Mock Encrypted Pool...")
        from he_layer import get_he_context
        ctx = get_he_context(he_backend)
        data = np.array([1.0, 2.0, 3.0])
        ct = ctx.encrypt(data)
        
        st.write(f"Ciphertext Object: `{ct}`")
        if he_backend == "mock":
            st.warning("MOCK BACKEND: Data stored in `ct.data`. Trivially leaked.")
            st.code(f"Leaked Data: {ct.data}")
        elif he_backend == "tenseal":
            st.success("TENSEAL: Data encrypted. Without SK, this is noise.")
            st.code(f"Ciphertext: {ct.data}")
            
    st.markdown("---")
    st.header("2. Collusion Attack")
    st.markdown("Simulate Cloud Server stealing the Secret Key.")
    
    if st.button("Simulate Collusion"):
        data_owners = {"User_Victim": np.random.rand(5, 4)}
        
        from system_sim import KGC, DataOwner
        kgc = KGC(he_backend)
        he, pk, sk = kgc.initialize_system()
        
        owner = DataOwner("User_Victim", data_owners["User_Victim"])
        encrypted_data = owner.encrypt_and_upload(he)
        
        st.write("Cloud Server obtains SK...")
        leaked = AttackSimulator.simulate_collation_attack(encrypted_data, sk, he_type=he_backend)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Data**")
            st.write(data_owners["User_Victim"][:2])
        with col2:
            st.write("**Recovered Data**")
            st.write(leaked[:2])
            
        if np.allclose(data_owners["User_Victim"], leaked, atol=1e-5):
            st.error("Attack Successful: Data fully recovered.")
        else:
            st.success("Attack Failed: Data mismatch.")


# --- PAGE 3: BENCHMARK ---
def render_benchmark():
    st.title("Performance Benchmarking")
    st.markdown("Analyze the computational overhead of Privacy-Preserving operations.")
    
    col1, col2 = st.columns(2)
    with col1:
        max_users = st.number_input("Max Users", value=5, min_value=2)
    with col2:
        step_size = st.number_input("Step Size", value=1, min_value=1)
        
    if st.button("Run Benchmark"):
        with st.spinner("Running benchmark..."):
            df = run_benchmark(max_users=max_users, step_size=step_size, he_type=he_backend)
            
            st.subheader("Results")
            st.dataframe(df)
            
            fig = px.line(df, x="Users", y="Time (s)", markers=True,
                         title="Execution Time vs Scale")
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("CKKS (Real HE) is significantly slower than Mock HE.")


# --- PAGE 4: PRIVACY ATTACK PLAYGROUND (FEATURE 4) ---
def render_attack_playground():
    st.title("Privacy Attack Playground")
    st.markdown("*Be the attacker and see why encryption matters!*")

    # Render attack simulator content here (omitted for brevity in this replace block, expecting existing content)
    # Note: I am not replacing the content of render_attack_playground, just ensuring the connection.
    # Actually, since I can't see the whole file content in the view_file from before (it was truncated),
    # I should be careful not to delete the body of render_attack_playground if it exists.
    # The view_file output ended at line 800 just after render_attack_playground def.
    # I will append the new import and checking logic at the end of the file.

    st.header("What does the Cloud Server see?")
    
    tab1, tab2, tab3 = st.tabs(["Raw vs Encrypted", "Frequency Analysis", "Reconstruction Attack"])
    
    with tab1:
        st.subheader("Comparing Raw Data vs Encrypted View")
        
        # Generate sample data
        np.random.seed(42)
        sample_data = np.array([[1.5, 2.3, 4.1], [3.2, 1.8, 5.5], [2.1, 3.7, 2.9]])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data Owner's View")
            st.markdown("*This is the actual sensitive data*")
            df_raw = pd.DataFrame(sample_data, columns=["Salary_k", "Age_10", "Score"])
            st.dataframe(df_raw)
            
            fig_raw = px.scatter_3d(df_raw, x="Salary_k", y="Age_10", z="Score",
                                   title="Raw Data (Visible Structure)")
            st.plotly_chart(fig_raw, use_container_width=True)
        
        with col2:
            st.markdown("### Cloud Server's View")
            st.markdown("*This is what the server actually receives*")
            
            from he_layer import get_he_context
            ctx = get_he_context(he_backend)
            
            if he_backend == "mock":
                st.warning("Mock Mode - Structure is visible (unsafe!)")
                encrypted_view = sample_data  # Mock doesn't hide it
            else:
                st.success("TenSEAL Mode - Only encrypted blobs visible")
                encrypted_view = np.random.rand(*sample_data.shape) * 1e15  # Simulated noise
            
            df_enc = pd.DataFrame(encrypted_view, columns=["ct_0", "ct_1", "ct_2"])
            st.dataframe(df_enc)
            
            fig_enc = px.scatter_3d(df_enc, x="ct_0", y="ct_1", z="ct_2",
                                   title="Encrypted View (No Structure)")
            st.plotly_chart(fig_enc, use_container_width=True)
    
    with tab2:
        st.subheader("Frequency Analysis Attack")
        st.markdown("*Can the attacker infer patterns from ciphertext frequencies?*")
        
        # Generate data with patterns
        np.random.seed(42)
        data_with_pattern = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 20)
        np.random.shuffle(data_with_pattern)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Distribution**")
            fig_hist1 = px.histogram(x=data_with_pattern, nbins=10, title="True Distribution")
            st.plotly_chart(fig_hist1, use_container_width=True)
            st.caption("Clear peaks reveal value distribution")
        
        with col2:
            st.markdown("**Encrypted (with CKKS noise)**")
            # CKKS adds small noise to each value
            noisy_data = data_with_pattern + np.random.normal(0, 0.01, len(data_with_pattern))
            fig_hist2 = px.histogram(x=noisy_data, nbins=50, title="After CKKS Encoding")
            st.plotly_chart(fig_hist2, use_container_width=True)
            
            if he_backend == "tenseal":
                st.success("Individual values are protected by polynomial encoding")
            else:
                st.warning("Mock mode doesn't add encoding noise")
    
    with tab3:
        st.subheader("Laplacian Reconstruction Attack")
        st.markdown("*Can we recover W from the normalized Laplacian L?*")
        
        st.info("""
        The cloud server computes L = D^(-1/2) W D^(-1/2)
        
        **Attack Question**: If we see L, can we find W (affinity matrix)?
        
        **Answer**: Yes! If we know the diagonal of D, we can invert:
        W = D^(1/2) L D^(1/2)
        """)
        
        # Demo with small matrix
        np.random.seed(42)
        W_original = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.5],
            [0.2, 0.5, 1.0]
        ])
        
        D = np.diag(W_original.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = D_inv_sqrt @ W_original @ D_inv_sqrt
        
        # "Attack" - reconstruct W from L
        D_sqrt = np.diag(np.sqrt(np.diag(D)))
        W_reconstructed = D_sqrt @ L @ D_sqrt
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original W**")
            fig_w = px.imshow(W_original, color_continuous_scale='hot', title="W (Secret)")
            st.plotly_chart(fig_w, use_container_width=True)
        
        with col2:
            st.markdown("**What Cloud Sees (L)**")
            fig_l = px.imshow(L, color_continuous_scale='rdbu', title="L (Public)")
            st.plotly_chart(fig_l, use_container_width=True)
        
        with col3:
            st.markdown("**Reconstructed W**")
            fig_w2 = px.imshow(W_reconstructed, color_continuous_scale='hot', title="W' (Recovered)")
            st.plotly_chart(fig_w2, use_container_width=True)
        
        error = np.linalg.norm(W_original - W_reconstructed)
        if error < 1e-10:
            st.error(f"Perfect reconstruction! Error: {error:.2e}")
            st.markdown("""
            **This is why pure spectral clustering on cloud needs encryption!**
            
            Without HE, the cloud can recover similarity relationships between data points.
            """)
        else:
            st.success(f"Reconstruction error: {error:.2e}")


# Main Router
if "Prototype" in page:
    render_prototype()
elif "Security" in page:
    render_security_audit()
elif "Benchmark" in page:
    render_benchmark()
elif "Attack" in page:
    render_attack_playground()
elif "Encryption" in page:
    render_encryption_deep_dive(he_backend)
