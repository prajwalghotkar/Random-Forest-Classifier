"""
Utility functions for Random Forest Classifier
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_sample_data(dataset_name, n_samples=1000, n_features=20, n_classes=2):
    """
    Load sample dataset based on user selection
    """
    try:
        if dataset_name == "Synthetic Binary Classification":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.7),
                n_redundant=int(n_features * 0.2),
                n_classes=2,
                random_state=42
            )
            feature_names = [f"Feature {i}" for i in range(n_features)]
            target_names = ["Class 0", "Class 1"]
            
        elif dataset_name == "Synthetic Multi-class":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.6),
                n_redundant=int(n_features * 0.2),
                n_classes=n_classes,
                n_clusters_per_class=1,
                random_state=42
            )
            feature_names = [f"Feature {i}" for i in range(n_features)]
            target_names = [f"Class {i}" for i in range(n_classes)]
            
        elif dataset_name == "Iris Dataset":
            data = load_iris()
            X, y = data.data, data.target
            feature_names = data.feature_names
            target_names = data.target_names
            
        elif dataset_name == "Breast Cancer Dataset":
            data = load_breast_cancer()
            X, y = data.data, data.target
            feature_names = data.feature_names
            target_names = data.target_names
            
        elif dataset_name == "Wine Dataset":
            data = load_wine()
            X, y = data.data, data.target
            feature_names = data.feature_names
            target_names = data.target_names
        
        return X, y, feature_names, target_names
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        # Return default synthetic data as fallback
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        feature_names = [f"Feature {i}" for i in range(10)]
        target_names = ["Class 0", "Class 1"]
        return X, y, feature_names, target_names

@st.cache_data
def split_data(X, y, test_size, random_state):
    """
    Split data into train and test sets
    """
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except:
        # If stratify fails (e.g., not enough samples per class), split without stratify
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_random_forest(X_train, y_train, n_estimators, max_features, bootstrap, max_samples, random_state):
    """
    Train Random Forest classifier with given hyperparameters
    """
    # Handle max_samples when bootstrap is False
    if not bootstrap:
        max_samples = None
    
    # Handle max_features conversion
    if max_features == "None":
        max_features = None
    elif max_features == "sqrt":
        max_features = "sqrt"
    elif max_features == "log2":
        max_features = "log2"
    else:
        try:
            max_features = float(max_features)
        except:
            max_features = "sqrt"
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        bootstrap=bootstrap,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get probability predictions
    if len(np.unique(y_test)) == 2:
        y_test_proba = model.predict_proba(X_test)
    else:
        y_test_proba = model.predict_proba(X_test)
    
    # Accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Cross-validation score
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(np.unique(y_train))))
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except:
        cv_mean = 0
        cv_std = 0
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Classification report
    try:
        report = classification_report(y_test, y_test_pred, output_dict=True)
    except:
        report = {}
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

def plot_confusion_matrix(cm, target_names):
    """
    Create confusion matrix plot using plotly
    """
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=target_names,
        y=target_names,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=500,
        height=500
    )
    
    return fig

def plot_roc_curve(y_test, y_test_proba, n_classes, target_names):
    """
    Plot ROC curves for binary or multi-class classification
    """
    fig = go.Figure()
    
    try:
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC curve (AUC = {roc_auc:.2f})',
                line=dict(color='darkorange', width=2)
            ))
        else:
            # Multi-class: One-vs-Rest
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            
            for i in range(min(n_classes, y_test_proba.shape[1])):
                if i < y_test_bin.shape[1]:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{target_names[i]} (AUC = {roc_auc:.2f})',
                        line=dict(width=2)
                    ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500,
            legend=dict(x=0.6, y=0.2)
        )
        
    except Exception as e:
        fig.add_annotation(text=f"Error generating ROC curve: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance using plotly
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    # Get top 15 features or all if less
    n_features = min(15, len(feature_names))
    top_indices = indices[:n_features]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = importance[top_indices]
    
    fig = go.Figure(data=go.Bar(
        x=top_importance[::-1],
        y=top_features[::-1],
        orientation='h',
        marker=dict(color=top_importance[::-1], colorscale='Viridis')
    ))
    
    fig.update_layout(
        title=f'Top {n_features} Feature Importances',
        xaxis_title='Importance',
        yaxis_title='Features',
        width=600,
        height=400
    )
    
    return fig

def plot_hyperparameter_comparison(results_df):
    """
    Create hyperparameter comparison plots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Effect of n_estimators', 'Effect of max_features',
                       'Effect of bootstrap', 'Effect of max_samples'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    try:
        # n_estimators effect
        if 'n_estimators' in results_df.columns and 'test_accuracy' in results_df.columns:
            n_est_means = results_df.groupby('n_estimators')['test_accuracy'].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=n_est_means['n_estimators'], y=n_est_means['test_accuracy'],
                          mode='lines+markers', name='n_estimators', line=dict(color='blue')),
                row=1, col=1
            )
        
        # max_features effect
        if 'max_features' in results_df.columns and 'test_accuracy' in results_df.columns:
            max_feat_means = results_df.groupby('max_features')['test_accuracy'].mean().reset_index()
            fig.add_trace(
                go.Bar(x=max_feat_means['max_features'], y=max_feat_means['test_accuracy'],
                       name='max_features', marker_color='orange'),
                row=1, col=2
            )
        
        # bootstrap effect
        if 'bootstrap' in results_df.columns and 'test_accuracy' in results_df.columns:
            bootstrap_means = results_df.groupby('bootstrap')['test_accuracy'].mean().reset_index()
            bootstrap_means['bootstrap'] = bootstrap_means['bootstrap'].astype(str)
            fig.add_trace(
                go.Bar(x=bootstrap_means['bootstrap'], y=bootstrap_means['test_accuracy'],
                       name='bootstrap', marker_color='green'),
                row=2, col=1
            )
        
        # max_samples effect (only for bootstrap=True)
        if 'bootstrap' in results_df.columns and 'max_samples' in results_df.columns and 'test_accuracy' in results_df.columns:
            bootstrap_true = results_df[results_df['bootstrap'] == True]
            if not bootstrap_true.empty:
                max_samp_means = bootstrap_true.groupby('max_samples')['test_accuracy'].mean().reset_index()
                max_samp_means['max_samples'] = max_samp_means['max_samples'].astype(str)
                fig.add_trace(
                    go.Bar(x=max_samp_means['max_samples'], y=max_samp_means['test_accuracy'],
                           name='max_samples', marker_color='red'),
                    row=2, col=2
                )
                
    except Exception as e:
        fig.add_annotation(text=f"Error generating comparison: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    fig.update_layout(height=700, showlegend=False, title_text="Hyperparameter Impact Analysis")
    return fig