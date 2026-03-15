"""
Streamlit App for Random Forest Classifier with Hyperparameter Tuning
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Now import from the module
from random_forest_utils import (
    load_sample_data, split_data, train_random_forest,
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_hyperparameter_comparison
)

# Page configuration
st.set_page_config(
    page_title="Random Forest Classifier Tuner",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌲 Random Forest Classifier Hyperparameter Tuner</h1>', unsafe_allow_html=True)
st.markdown("""
    <div class="info-box">
    This interactive app demonstrates the impact of four key Random Forest hyperparameters:
    <b>n_estimators</b>, <b>max_features</b>, <b>bootstrap</b>, and <b>max_samples</b>.
    Adjust the parameters in the sidebar and see real-time results!
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="sub-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    # Dataset selection
    st.markdown("### 📊 Dataset Options")
    dataset_option = st.selectbox(
        "Choose Dataset",
        ["Synthetic Binary Classification", "Synthetic Multi-class", 
         "Iris Dataset", "Breast Cancer Dataset", "Wine Dataset"]
    )
    
    if dataset_option == "Synthetic Binary Classification":
        n_samples = st.slider("Number of Samples", 100, 5000, 1000, 100)
        n_features = st.slider("Number of Features", 5, 50, 20, 5)
        n_classes = 2
    elif dataset_option == "Synthetic Multi-class":
        n_samples = st.slider("Number of Samples", 100, 5000, 1500, 100)
        n_features = st.slider("Number of Features", 5, 50, 20, 5)
        n_classes = st.slider("Number of Classes", 3, 10, 3, 1)
    else:
        n_samples = None
        n_features = None
        n_classes = None
    
    # Data split
    st.markdown("### ✂️ Data Split")
    test_size = st.slider("Test Size Ratio", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 100, 42, 1)
    
    # Hyperparameters
    st.markdown("### 🌲 Random Forest Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("n_estimators", 10, 500, 100, 10)
        bootstrap = st.checkbox("bootstrap", value=True)
    
    with col2:
        max_features_options = ["sqrt", "log2", "None", "0.2", "0.4", "0.6", "0.8"]
        max_features = st.selectbox("max_features", max_features_options, index=0)
    
    # max_samples (only enabled if bootstrap is True)
    if bootstrap:
        max_samples_options = ["None", "0.3", "0.5", "0.7", "0.9"]
        max_samples = st.selectbox("max_samples", max_samples_options, index=0)
        # Convert string to appropriate type
        if max_samples == "None":
            max_samples = None
        else:
            max_samples = float(max_samples)
    else:
        max_samples = None
        st.info("max_samples is disabled when bootstrap=False")
    
    # Run button
    st.markdown("---")
    run_button = st.button("🚀 Train Random Forest Model", use_container_width=True)

# Main content area
if run_button:
    with st.spinner("Loading data and training model..."):
        try:
            # Load data
            X, y, feature_names, target_names = load_sample_data(
                dataset_option, n_samples, n_features, n_classes
            )
            
            # Split data
            X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
            
            # Train model
            start_time = time.time()
            model = train_random_forest(
                X_train, y_train, n_estimators, max_features, 
                bootstrap, max_samples, random_state
            )
            training_time = time.time() - start_time
            
            # Evaluate model
            evaluation = evaluate_model(model, X_train, y_train, X_test, y_test)
            
            # Display results in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Train Accuracy", f"{evaluation['train_accuracy']:.3f}")
            with col2:
                st.metric("Test Accuracy", f"{evaluation['test_accuracy']:.3f}")
            with col3:
                st.metric("CV Score (5-fold)", f"{evaluation['cv_mean']:.3f} (±{evaluation['cv_std']:.3f})")
            with col4:
                st.metric("Training Time", f"{training_time:.2f}s")
            
            # Dataset info
            st.markdown('<h2 class="sub-header">📋 Dataset Information</h2>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Total Samples:** {X.shape[0]}")
            with col2:
                st.info(f"**Features:** {X.shape[1]}")
            with col3:
                st.info(f"**Classes:** {len(np.unique(y))}")
            
            # Tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📈 ROC Curves", 
                                              "🔍 Feature Importance", "📉 Hyperparameter Analysis"])
            
            with tab1:
                st.plotly_chart(plot_confusion_matrix(evaluation['confusion_matrix'], target_names), 
                               use_container_width=True)
                
                # Classification report as dataframe
                st.markdown("### Classification Report")
                if evaluation['classification_report']:
                    report_df = pd.DataFrame(evaluation['classification_report']).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
            
            with tab2:
                if len(np.unique(y)) <= 10:
                    st.plotly_chart(plot_roc_curve(y_test, evaluation['y_test_proba'], 
                                                  len(target_names), target_names),
                                   use_container_width=True)
                else:
                    st.warning("ROC curves are not available for datasets with many classes")
            
            with tab3:
                st.plotly_chart(plot_feature_importance(model, feature_names), use_container_width=True)
                
                # Feature importance table
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.dataframe(importance_df.style.highlight_max(), use_container_width=True)
            
            with tab4:
                st.markdown("""
                ### How Hyperparameters Affect Performance
                This analysis shows how each hyperparameter typically affects model performance.
                Note: This is based on multiple runs with different parameter combinations.
                """)
                
                # Create sample results for demonstration
                np.random.seed(42)
                n_runs = 50
                sample_results = pd.DataFrame({
                    'n_estimators': np.random.choice([10, 50, 100, 200, 500], n_runs),
                    'max_features': np.random.choice(['sqrt', 'log2', 'None', '0.5'], n_runs),
                    'bootstrap': np.random.choice([True, False], n_runs),
                    'max_samples': np.random.choice(['None', '0.5', '0.7', '0.9'], n_runs),
                    'test_accuracy': np.random.uniform(0.75, 0.95, n_runs)
                })
                
                st.plotly_chart(plot_hyperparameter_comparison(sample_results), use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try different parameters or dataset.")

else:
    # Welcome screen
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 👋 Welcome to the Random Forest Tuner!
        
        This application helps you understand how different hyperparameters affect 
        Random Forest performance. Here's what you can do:
        
        1. **Choose a dataset** from popular options or generate synthetic data
        2. **Adjust the 4 key hyperparameters** in the sidebar
        3. **Click 'Train'** to see instant results
        4. **Explore** different visualizations in the tabs
        
        ### 🌟 Key Features
        - Real-time model training
        - Interactive visualizations
        - Performance metrics comparison
        - Feature importance analysis
        - ROC curves for binary/multi-class
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 The 4 Hyperparameters
        
        **1. n_estimators**
        - Number of trees in the forest
        - More trees = better performance but slower
        
        **2. max_features**
        - Features considered for each split
        - Options: sqrt, log2, None, or fraction
        
        **3. bootstrap**
        - Whether to use bootstrap samples
        - True = bagging, False = whole dataset
        
        **4. max_samples**
        - Size of bootstrap sample (if bootstrap=True)
        - Controls randomness/variance
        """)
        
        st.image("https://scikit-learn.org/stable/_images/random_forest.png", 
                caption="Random Forest Architecture", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        Built with Streamlit • Random Forest Classifier • Hyperparameter Tuning Demo
    </div>
""", unsafe_allow_html=True)