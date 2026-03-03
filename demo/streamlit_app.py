"""Streamlit demo for SHAP XAI project."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.device import set_seed
from src.data.loader import DataLoader
from src.models.manager import ModelManager
from src.explainers.shap_explainer import SHAPExplainer
from src.metrics.xai_metrics import XAIMetrics

# Page config
st.set_page_config(
    page_title="SHAP XAI Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p>This demo is for <strong>research and educational purposes only</strong>. 
    XAI outputs may be unstable or misleading and should NOT be used for regulated 
    decisions without human review. Always validate explanations with domain experts.</p>
</div>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧠 SHAP Values for Model Interpretation</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")

# Load configuration
config = load_config()

# Dataset selection
dataset_options = ["iris", "wine", "breast_cancer", "synthetic"]
selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    dataset_options,
    index=dataset_options.index(config.data.dataset_name)
)

# Model selection
model_options = ["random_forest", "logistic_regression", "decision_tree"]
selected_model = st.sidebar.selectbox(
    "Select Model",
    model_options,
    index=model_options.index(config.model.model_type)
)

# SHAP explainer selection
explainer_options = ["tree", "kernel", "deep"]
selected_explainer = st.sidebar.selectbox(
    "Select SHAP Explainer",
    explainer_options,
    index=explainer_options.index(config.shap.explainer_type)
)

# Update config
config.data.dataset_name = selected_dataset
config.model.model_type = selected_model
config.shap.explainer_type = selected_explainer

# Random seed
random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=10000)

# Set seed
set_seed(random_seed)

# Main content
if st.sidebar.button("Run Analysis", type="primary"):
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components
        status_text.text("Initializing components...")
        progress_bar.progress(10)
        
        data_loader = DataLoader(config)
        model_manager = ModelManager(config)
        shap_explainer = SHAPExplainer(config)
        xai_metrics = XAIMetrics(config)
        
        # Load data
        status_text.text("Loading dataset...")
        progress_bar.progress(20)
        
        X, y, feature_names, target_names = data_loader.load_dataset()
        X_processed, y_processed = data_loader.preprocess_data(X, y)
        X_train, X_test, y_train, y_test = data_loader.split_data(X_processed, y_processed)
        
        # Update feature names
        data_loader.feature_names = feature_names
        data_loader.target_names = target_names
        
        # Train model
        status_text.text("Training model...")
        progress_bar.progress(40)
        
        model = model_manager.train_model(X_train, y_train)
        model_metrics = model_manager.evaluate_model(X_test, y_test)
        
        # Create explainer
        status_text.text("Creating SHAP explainer...")
        progress_bar.progress(60)
        
        explainer = shap_explainer.create_explainer(model, X_train, feature_names)
        shap_values = shap_explainer.compute_shap_values(X_test)
        
        # Evaluate explanations
        status_text.text("Evaluating explanations...")
        progress_bar.progress(80)
        
        evaluation_metrics = xai_metrics.compute_comprehensive_metrics(
            model, X_test, y_test, shap_values, feature_names
        )
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Store results in session state
        st.session_state['model'] = model
        st.session_state['shap_explainer'] = shap_explainer
        st.session_state['shap_values'] = shap_values
        st.session_state['feature_names'] = feature_names
        st.session_state['target_names'] = target_names
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['model_metrics'] = model_metrics
        st.session_state['evaluation_metrics'] = evaluation_metrics
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.stop()

# Display results if analysis is complete
if 'model' in st.session_state:
    
    # Model Performance
    st.header("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{st.session_state['model_metrics']['accuracy']:.4f}")
    
    with col2:
        st.metric("Dataset", config.data.dataset_name.title())
    
    with col3:
        st.metric("Model", config.model.model_type.replace('_', ' ').title())
    
    with col4:
        st.metric("Explainer", config.shap.explainer_type.title())
    
    # Feature Importance
    st.header("🎯 Feature Importance")
    
    feature_importance = st.session_state['shap_explainer'].get_feature_importance()
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame([
        {"Feature": feature, "Importance": importance}
        for feature, importance in feature_importance.items()
    ]).sort_values("Importance", ascending=True)
    
    # Plotly bar chart
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance (Mean |SHAP value|)",
        color="Importance",
        color_continuous_scale="viridis"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Local Explanations
    st.header("🔍 Local Explanations")
    
    # Instance selection
    instance_idx = st.selectbox(
        "Select Test Instance",
        range(len(st.session_state['X_test'])),
        format_func=lambda x: f"Instance {x} (True: {st.session_state['target_names'][st.session_state['y_test'][x]]})"
    )
    
    # Get explanation for selected instance
    explanation = st.session_state['shap_explainer'].explain_prediction(
        st.session_state['X_test'][instance_idx]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Instance Values")
        instance_df = pd.DataFrame({
            "Feature": st.session_state['feature_names'],
            "Value": st.session_state['X_test'][instance_idx]
        })
        st.dataframe(instance_df, use_container_width=True)
    
    with col2:
        st.subheader("SHAP Values")
        shap_df = pd.DataFrame({
            "Feature": st.session_state['feature_names'],
            "SHAP Value": explanation["shap_values"]
        }).sort_values("SHAP Value", key=abs, ascending=False)
        st.dataframe(shap_df, use_container_width=True)
    
    # SHAP Values Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort features by absolute SHAP value
    sorted_indices = np.argsort(np.abs(explanation["shap_values"]))[::-1]
    sorted_features = [st.session_state['feature_names'][i] for i in sorted_indices]
    sorted_values = [explanation["shap_values"][i] for i in sorted_indices]
    
    # Color bars based on positive/negative
    colors = ['red' if v < 0 else 'blue' for v in sorted_values]
    
    ax.barh(range(len(sorted_features)), sorted_values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('SHAP Value')
    ax.set_title(f'SHAP Values for Instance {instance_idx}')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    st.pyplot(fig)
    
    # Evaluation Metrics
    st.header("📈 Explanation Evaluation")
    
    eval_metrics = st.session_state['evaluation_metrics']
    
    if "faithfulness_deletion" in eval_metrics:
        st.subheader("Faithfulness Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Deletion Test**")
            deletion_data = eval_metrics["faithfulness_deletion"]
            deletion_df = pd.DataFrame({
                "Percentage": deletion_data["deletion_percentages"],
                "Score": deletion_data["deletion_scores"]
            })
            st.dataframe(deletion_df, use_container_width=True)
        
        with col2:
            st.write("**Insertion Test**")
            insertion_data = eval_metrics["faithfulness_insertion"]
            insertion_df = pd.DataFrame({
                "Percentage": insertion_data["insertion_percentages"],
                "Score": insertion_data["insertion_scores"]
            })
            st.dataframe(insertion_df, use_container_width=True)
    
    # Explanation Statistics
    if "explanation_stats" in eval_metrics:
        st.subheader("Explanation Statistics")
        
        stats = eval_metrics["explanation_stats"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean |SHAP|", f"{stats['mean_abs_shap']:.4f}")
        
        with col2:
            st.metric("Std |SHAP|", f"{stats['std_abs_shap']:.4f}")
        
        with col3:
            st.metric("Max |SHAP|", f"{stats['max_abs_shap']:.4f}")
        
        with col4:
            st.metric("Min |SHAP|", f"{stats['min_abs_shap']:.4f}")
    
    # Dataset Information
    st.header("📋 Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Dataset**: {config.data.dataset_name.title()}")
        st.write(f"**Features**: {len(st.session_state['feature_names'])}")
        st.write(f"**Classes**: {len(st.session_state['target_names'])}")
        st.write(f"**Test Samples**: {len(st.session_state['X_test'])}")
    
    with col2:
        st.write("**Feature Names**:")
        for i, feature in enumerate(st.session_state['feature_names']):
            st.write(f"  {i+1}. {feature}")
        
        st.write("**Target Names**:")
        for i, target in enumerate(st.session_state['target_names']):
            st.write(f"  {i+1}. {target}")

else:
    st.info("👈 Please configure the settings in the sidebar and click 'Run Analysis' to start.")
    
    # Show example of what the demo will do
    st.header("🎯 What This Demo Does")
    
    st.markdown("""
    This interactive demo showcases SHAP (SHapley Additive exPlanations) values for model interpretation:
    
    1. **Data Loading**: Loads datasets (Iris, Wine, Breast Cancer, or synthetic data)
    2. **Model Training**: Trains various models (Random Forest, Logistic Regression, Decision Tree)
    3. **SHAP Explanations**: Computes SHAP values using different explainers (Tree, Kernel, Deep)
    4. **Visualization**: Creates interactive plots for feature importance and local explanations
    5. **Evaluation**: Tests explanation faithfulness using deletion and insertion tests
    
    **Key Features**:
    - Interactive instance selection for local explanations
    - Real-time SHAP value computation and visualization
    - Comprehensive evaluation metrics
    - Multiple model and explainer options
    - Educational focus with clear explanations
    """)
    
    st.header("⚠️ Important Notes")
    
    st.markdown("""
    - **Research Only**: This demo is for educational and research purposes
    - **Explanation Instability**: SHAP values may vary across different runs
    - **Model Limitations**: Results depend on model quality and data distribution
    - **Human Validation**: Always validate explanations with domain experts
    - **Not for Production**: Do not use for regulated decisions without review
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>SHAP XAI Demo - For Research and Educational Purposes Only</p>
    <p>⚠️ Always validate explanations with domain experts</p>
</div>
""", unsafe_allow_html=True)
