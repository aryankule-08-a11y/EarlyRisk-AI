"""
📈 Visualizations Page
Comprehensive charts and graphs for fraud analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="📈 Visualizations", page_icon="📈", layout="wide")

def main():
    st.markdown("<h1 style='text-align:center;color:#e94560;'>📈 Visualizations</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#a8a8a8;'>Interactive charts and analytics dashboard</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.get('weak_labeled_data') is None:
        st.warning("⚠️ No data available! Complete previous steps first.")
        return
    
    df = st.session_state.weak_labeled_data
    
    # Get column names
    amount_col = next((c for c in ['amount', 'transaction_amount'] if c in df.columns), None)
    user_col = next((c for c in ['user_id', 'customer_id'] if c in df.columns), None)
    timestamp_col = next((c for c in ['timestamp', 'date', 'datetime'] if c in df.columns), None)
    
    # Chart selection
    # Chart selection
    chart_type = st.selectbox("Select Visualization", [
        "🎯 Fraud vs Non-Fraud Distribution",
        "💰 Transaction Amount Analysis",
        "📏 Rule Activation Frequency",
        "📊 Feature Distributions",
        "🌙 Time-based Analysis",
        "🗺️ Geographic Analysis",
        "🔍 Feature Importance (if model trained)",
        "🧠 SHAP Global Explainability",
        "📈 ROC Curve Comparison"
    ])
    
    st.markdown("---")
    
    if chart_type == "🎯 Fraud vs Non-Fraud Distribution":
        cols = st.columns(2)
        
        with cols[0]:
            label_counts = df['weak_label'].value_counts()
            fig = px.pie(values=label_counts.values, names=['Non-Fraud', 'Fraud'],
                        color_discrete_sequence=['#4ade80', '#e94560'],
                        title='Fraud Label Distribution', hole=0.4)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
            st.plotly_chart(fig, use_container_width=True)
        
        with cols[1]:
            fig = px.bar(x=['Non-Fraud', 'Fraud'], y=label_counts.values,
                        color=['Non-Fraud', 'Fraud'], color_discrete_map={'Non-Fraud': '#4ade80', 'Fraud': '#e94560'},
                        title='Fraud Count Comparison')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8',
                            showlegend=False, xaxis_title='', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        fraud_pct = df['weak_label'].mean() * 100
        st.metric("Fraud Rate", f"{fraud_pct:.2f}%", delta=f"{df['weak_label'].sum()} transactions")
    
    elif chart_type == "💰 Transaction Amount Analysis":
        if amount_col:
            cols = st.columns(2)
            
            with cols[0]:
                fig = px.histogram(df, x=amount_col, color='weak_label',
                                  color_discrete_map={0: '#4ade80', 1: '#e94560'},
                                  title='Amount Distribution by Label', barmode='overlay', nbins=50)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
                st.plotly_chart(fig, use_container_width=True)
            
            with cols[1]:
                fig = px.box(df, x='weak_label', y=amount_col, color='weak_label',
                            color_discrete_map={0: '#4ade80', 1: '#e94560'},
                            title='Amount by Fraud Label')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
                st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Non-Fraud Amount", f"${df[df['weak_label']==0][amount_col].mean():.2f}")
            with col2:
                st.metric("Avg Fraud Amount", f"${df[df['weak_label']==1][amount_col].mean():.2f}")
            with col3:
                ratio = df[df['weak_label']==1][amount_col].mean() / max(df[df['weak_label']==0][amount_col].mean(), 1)
                st.metric("Amount Ratio", f"{ratio:.2f}x")
        else:
            st.info("Amount column not found")
    
    elif chart_type == "📏 Rule Activation Frequency":
        rule_cols = ['rule_amount', 'rule_time', 'rule_country', 'rule_velocity']
        existing_rules = [c for c in rule_cols if c in df.columns]
        
        if existing_rules:
            rule_counts = {c.replace('rule_', '').title(): df[c].sum() for c in existing_rules}
            
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'pie'}]])
            
            fig.add_trace(go.Bar(x=list(rule_counts.keys()), y=list(rule_counts.values()),
                                marker_color=['#e94560', '#533483', '#4ade80', '#60a5fa'][:len(rule_counts)]),
                         row=1, col=1)
            
            fig.add_trace(go.Pie(labels=list(rule_counts.keys()), values=list(rule_counts.values()),
                                marker_colors=['#e94560', '#533483', '#4ade80', '#60a5fa'][:len(rule_counts)]),
                         row=1, col=2)
            
            fig.update_layout(title='Rule Activation Analysis', height=400,
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap of rule combinations
            st.markdown("### 🔗 Rule Combinations")
            rule_df = df[existing_rules]
            combinations = rule_df.groupby(existing_rules).size().reset_index(name='count')
            combinations = combinations.sort_values('count', ascending=False).head(10)
            st.dataframe(combinations, use_container_width=True)
        else:
            st.info("Rule columns not found")
    
    elif chart_type == "📊 Feature Distributions":
        feature_cols = ['transaction_time_diff', 'amount_deviation', 'transactions_last_5min', 
                       'new_country_flag', 'night_transaction_flag']
        existing = [c for c in feature_cols if c in df.columns]
        
        selected = st.selectbox("Select Feature", existing if existing else ['No features'])
        
        if selected and selected in df.columns:
            cols = st.columns(2)
            
            with cols[0]:
                fig = px.histogram(df, x=selected, color='weak_label',
                                  color_discrete_map={0: '#4ade80', 1: '#e94560'},
                                  title=f'{selected} Distribution', barmode='overlay')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
                st.plotly_chart(fig, use_container_width=True)
            
            with cols[1]:
                fig = px.violin(df, y=selected, x='weak_label', color='weak_label',
                               color_discrete_map={0: '#4ade80', 1: '#e94560'},
                               title=f'{selected} by Label')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "🌙 Time-based Analysis":
        if 'transaction_hour' in df.columns:
            hourly = df.groupby(['transaction_hour', 'weak_label']).size().reset_index(name='count')
            
            fig = px.line(hourly, x='transaction_hour', y='count', color='weak_label',
                         color_discrete_map={0: '#4ade80', 1: '#e94560'},
                         title='Transactions by Hour', markers=True)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8',
                            xaxis_title='Hour of Day', yaxis_title='Transaction Count')
            st.plotly_chart(fig, use_container_width=True)
            
            # Night vs Day
            if 'night_transaction_flag' in df.columns:
                night_fraud = df[df['night_transaction_flag'] == 1]['weak_label'].mean()
                day_fraud = df[df['night_transaction_flag'] == 0]['weak_label'].mean()
                
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Night Fraud Rate", f"{night_fraud*100:.2f}%")
                with cols[1]:
                    st.metric("Day Fraud Rate", f"{day_fraud*100:.2f}%")
        else:
            st.info("Time data not available")
    
    elif chart_type == "🗺️ Geographic Analysis":
        country_col = next((c for c in ['country', 'location'] if c in df.columns), None)
        
        if country_col:
            geo_stats = df.groupby(country_col).agg({
                'weak_label': ['sum', 'count', 'mean']
            }).round(4)
            geo_stats.columns = ['Fraud Count', 'Total', 'Fraud Rate']
            geo_stats = geo_stats.sort_values('Fraud Count', ascending=False)
            
            cols = st.columns(2)
            
            with cols[0]:
                fig = px.bar(geo_stats.reset_index(), x=country_col, y='Fraud Count',
                            color='Fraud Rate', color_continuous_scale='Reds',
                            title='Fraud by Country')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
                st.plotly_chart(fig, use_container_width=True)
            
            with cols[1]:
                st.dataframe(geo_stats.style.background_gradient(cmap='Reds', subset=['Fraud Rate']), 
                           use_container_width=True)
        else:
            st.info("Country/location column not found")
    
    elif chart_type == "🔍 Feature Importance (if model trained)":
        if st.session_state.get('trained_models') and st.session_state.get('feature_cols'):
            model_name = st.session_state.get('selected_model', list(st.session_state.trained_models.keys())[0])
            model = st.session_state.trained_models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'Feature': st.session_state.feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='Reds',
                            title=f'Feature Importance - {model_name}')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8',
                                height=500)
                st.plotly_chart(fig, use_container_width=True)
            elif hasattr(model, 'coef_'):
                importance = pd.DataFrame({
                    'Feature': st.session_state.feature_cols,
                    'Coefficient': np.abs(model.coef_[0])
                }).sort_values('Coefficient', ascending=True)
                
                fig = px.bar(importance, x='Coefficient', y='Feature', orientation='h',
                            color='Coefficient', color_continuous_scale='Blues',
                            title=f'Feature Coefficients (Absolute) - {model_name}')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trained models available. Train a model first.")

    elif chart_type == "🧠 SHAP Global Explainability":
        if not st.session_state.get('trained_models') or 'X_test' not in st.session_state:
            st.warning("⚠️ Please train models first to see SHAP explanations.")
        else:
            import shap
            import matplotlib.pyplot as plt
            
            model_name = st.session_state.get('selected_model', list(st.session_state.trained_models.keys())[0])
            model = st.session_state.trained_models[model_name]
            X_test = st.session_state.X_test
            feature_cols = st.session_state.feature_cols
            
            st.markdown(f"### 🧠 SHAP Explanation for **{model_name}**")
            st.markdown("Global interpretability: How features impact the model overall.")
            
            with st.spinner("Generating global SHAP values (this may take a moment)..."):
                try:
                    # Sample data for speed if too large
                    if len(X_test) > 500:
                        X_sample = X_test[:500]
                    else:
                        X_sample = X_test
                        
                    # Calculate SHAP values
                    if hasattr(model, 'predict_proba'): # Estimators
                        if 'XGB' in str(type(model)) or 'RandomForest' in str(type(model)):
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer(X_sample)
                        else: # Linear or others
                            # Use random background
                            background = shap.maskers.Independent(st.session_state.X_train, max_samples=100)
                            explainer = shap.LinearExplainer(model, background)
                            shap_values = explainer(X_sample)
                            
                    # Get correct shape if needed
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] # Positive class
                    elif len(shap_values.shape) == 3:
                        shap_values = shap_values[:, :, 1]

                    # Summary Plot
                    st.markdown("#### Beehive Summary Plot")
                    fig, ax = plt.subplots()
                    shap.plots.beeswarm(shap_values, max_display=10, show=False)
                    st.pyplot(fig)
                    
                    st.markdown("#### Mean Feature Importance")
                    fig, ax = plt.subplots()
                    shap.plots.bar(shap_values, max_display=10, show=False)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Could not generate SHAP plots for this model: {str(e)}")
                    st.info("Try selecting a Tree-based model (Random Forest/XGBoost) for better SHAP support.")
    
    elif chart_type == "📈 ROC Curve Comparison":
        if st.session_state.get('model_metrics'):
            from sklearn.metrics import roc_curve
            
            fig = go.Figure()
            colors = ['#e94560', '#533483', '#4ade80', '#60a5fa']
            
            for i, (name, metrics) in enumerate(st.session_state.model_metrics.items()):
                fpr, tpr, _ = roc_curve(st.session_state.y_test, metrics['Probabilities'])
                auc = metrics['ROC-AUC']
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                        name=f'{name} (AUC={auc:.3f})',
                                        line=dict(color=colors[i % len(colors)])))
            
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                    name='Random', line=dict(dash='dash', color='gray')))
            
            fig.update_layout(title='ROC Curve Comparison',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trained models available. Train models first.")

if __name__ == "__main__":
    main()
