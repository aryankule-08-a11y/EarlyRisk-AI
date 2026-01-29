"""
🤖 Model Training Page
Train and evaluate ML models on weak supervision labels
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="🤖 Model Training", page_icon="🤖", layout="wide")

def prepare_features(df):
    """Prepare features for model training"""
    feature_cols = ['transaction_time_diff', 'amount_deviation', 'transactions_last_5min',
                   'new_country_flag', 'night_transaction_flag', 'is_weekend', 'user_transaction_count']
    
    # Add amount if available
    amount_col = next((c for c in ['amount', 'transaction_amount'] if c in df.columns), None)
    if amount_col:
        feature_cols.append(amount_col)
    
    existing_cols = [c for c in feature_cols if c in df.columns]
    X = df[existing_cols].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Replace inf values
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, existing_cols

@st.cache_resource
def train_models(X_train, y_train, selected_models):
    """Train selected models"""
    models = {}
    
    if 'Logistic Regression' in selected_models:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        models['Logistic Regression'] = lr
    
    if 'Random Forest' in selected_models:
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
    
    if 'XGBoost' in selected_models:
        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
            xgb.fit(X_train, y_train)
            models['XGBoost'] = xgb
        except ImportError:
            st.warning("XGBoost not installed. Skipping...")
    
    return models

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
        'Predictions': y_pred,
        'Probabilities': y_proba
    }

def main():
    st.markdown("<h1 style='text-align:center;color:#e94560;'>🤖 Model Training</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#a8a8a8;'>Train ML models on weak supervision labels</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.get('weak_labeled_data') is None:
        st.warning("⚠️ Weak labels not generated! Go to Weak Supervision Rules page first.")
        return
    
    df = st.session_state.weak_labeled_data
    
    # Prepare features
    X, feature_cols = prepare_features(df)
    y = df['weak_label']
    
    st.markdown("### 📊 Training Data Overview")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Samples", f"{len(df):,}")
    with cols[1]:
        st.metric("Features", len(feature_cols))
    with cols[2]:
        st.metric("Fraud Cases", f"{y.sum():,}")
    with cols[3]:
        st.metric("Fraud Rate", f"{y.mean()*100:.1f}%")
    
    st.markdown("**Features used:** " + ", ".join(feature_cols))
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        
    with col2:
        available_models = ['Logistic Regression', 'Random Forest', 'XGBoost']
        selected_models = st.multiselect("Select Models", available_models, default=['Logistic Regression', 'Random Forest'])
    
    if st.button("🚀 Train Models", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one model!")
            return
        
        with st.spinner("Training models..."):
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y if y.sum() >= 2 else None
            )
            
            # Store in session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
            st.session_state.feature_cols = feature_cols
            
            # Train models
            models = train_models(X_train, y_train, selected_models)
            st.session_state.trained_models = models
            
            # Evaluate models
            metrics = {}
            for name, model in models.items():
                metrics[name] = evaluate_model(model, X_test, y_test)
            st.session_state.model_metrics = metrics
            
            st.success(f"✅ Trained {len(models)} models successfully!")
    
    # Display results
    if st.session_state.get('model_metrics'):
        metrics = st.session_state.model_metrics
        
        st.markdown("---")
        st.markdown("### 📈 Model Performance")
        
        # Metrics comparison table
        metrics_df = pd.DataFrame({
            name: {k: v for k, v in m.items() if k not in ['Predictions', 'Probabilities']}
            for name, m in metrics.items()
        }).T
        
        st.dataframe(metrics_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=0), use_container_width=True)
        
        # Model selection
        st.markdown("---")
        st.markdown("### 🎯 Select Best Model")
        
        best_model = max(metrics.items(), key=lambda x: x[1]['F1-Score'])[0]
        selected = st.selectbox("Choose model for predictions", list(metrics.keys()), 
                               index=list(metrics.keys()).index(best_model))
        st.session_state.selected_model = selected
        
        st.success(f"✅ Selected: **{selected}**")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix", "📈 ROC Curve", "🔍 Feature Importance"])
        
        with tab1:
            y_test = st.session_state.y_test
            y_pred = metrics[selected]['Predictions']
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Non-Fraud', 'Fraud'], y=['Non-Fraud', 'Fraud'],
                           color_continuous_scale='Blues', text_auto=True)
            fig.update_layout(title=f"Confusion Matrix - {selected}",
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = go.Figure()
            for name, m in metrics.items():
                fpr, tpr, _ = roc_curve(st.session_state.y_test, m['Probabilities'])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={m['ROC-AUC']:.3f})", mode='lines'))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', line=dict(dash='dash', color='gray')))
            fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR",
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            model = st.session_state.trained_models[selected]
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'Feature': st.session_state.feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='Reds')
                fig.update_layout(title=f"Feature Importance - {selected}",
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
                st.plotly_chart(fig, use_container_width=True)
            elif hasattr(model, 'coef_'):
                importance = pd.DataFrame({
                    'Feature': st.session_state.feature_cols,
                    'Coefficient': model.coef_[0]
                }).sort_values('Coefficient', ascending=True)
                
                fig = px.bar(importance, x='Coefficient', y='Feature', orientation='h',
                            color='Coefficient', color_continuous_scale='RdBu')
                fig.update_layout(title=f"Feature Coefficients - {selected}",
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
                st.plotly_chart(fig, use_container_width=True)
        
        st.info("👉 Navigate to **Fraud Prediction** page to make predictions on new transactions.")

if __name__ == "__main__":
    main()
