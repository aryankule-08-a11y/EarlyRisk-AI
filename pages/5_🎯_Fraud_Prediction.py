"""
🎯 Fraud Prediction Page
Make real-time fraud predictions on new transactions with SHAP explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg') # Force headless backend for cloud deployment
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎯 Fraud Prediction", page_icon="🎯", layout="wide")

def predict_fraud(features, model, scaler):
    """Make fraud prediction"""
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    proba = model.predict_proba(X_scaled)[0][1]
    prediction = 1 if proba >= 0.5 else 0
    
    # Risk category
    if proba < 0.3:
        risk = "🟢 LOW RISK"
        color = "#4ade80"
    elif proba < 0.7:
        risk = "🟡 MEDIUM RISK"
        color = "#fbbf24"
    else:
        risk = "🔴 HIGH RISK"
        color = "#ef4444"
    
    return prediction, proba, risk, color, X_scaled

def check_rules(amount, user_avg, time_diff, transactions_5min, new_country, is_night):
    """Check which rules are triggered"""
    rules = []
    
    if amount > user_avg * 3:
        rules.append(("💰 High Amount", f"Amount ({amount:.2f}) > 3x User Avg ({user_avg:.2f})"))
    
    if time_diff < 2:
        rules.append(("⚡ Rapid Transaction", f"Time diff ({time_diff:.1f} min) < 2 minutes"))
    
    if new_country:
        rules.append(("🌍 New Country", "Transaction from new location"))
    
    if transactions_5min >= 3:
        rules.append(("🔄 High Velocity", f"{transactions_5min} transactions in last 5 min"))
    
    if is_night:
        rules.append(("🌙 Night Transaction", "Transaction during 12 AM - 6 AM"))
    
    return rules

def plot_shap_waterfall(model, X_single, feature_names, X_train=None):
    """Generate SHAP waterfall plot for a single prediction"""
    try:
        # Determine explainer type based on model
        model_type = type(model).__name__
        
        if 'XGBClassifier' in model_type or 'RandomForest' in model_type or 'Tree' in model_type:
             # Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_single)
        elif 'LogisticRegression' in model_type or 'Linear' in model_type:
            # Linear models
            # For linear models, we need a background dataset (X_train summary)
            if X_train is not None:
                # Use a small sample for background to speed up
                background = shap.maskers.Independent(X_train, max_samples=100)
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer(X_single)
            else:
                return None, "Training data needed for Linear SHAP"
        else:
            # Generic kernel explainer (slower)
            if X_train is not None:
                # Use kmeans to summarize background data
                background = shap.kmeans(X_train, 10)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                # Kernel explainer returns list for classification (class 0, class 1)
                shap_values = explainer.shap_values(X_single)
                # Format for plot - taking positive class
                explanation = shap.Explanation(values=shap_values[1][0], 
                                             base_values=explainer.expected_value[1], 
                                             data=X_single[0], 
                                             feature_names=feature_names)
                return explanation, "Kernel"
            else:
                return None, "Training data needed for Kernel SHAP"
        
        # For TreeExplainer, shap_values is already an Explanation object
        # but we need to ensure we get the explanation for the single instance
        if isinstance(shap_values, list): # Some old versions return list
             # If list, it's usually [class0, class1] for classifiers
             # We want class 1 (Fraud)
             sv = shap_values[1]
        elif len(shap_values.shape) == 2: # (1, n_features)
             sv = shap_values[0] # Single instance explanation
        elif len(shap_values.shape) == 3: # (1, n_features, n_classes) for some XGBoost versions
             sv = shap_values[0, :, 1]
        else:
             sv = shap_values[0]
            
        return sv, model_type
        
    except Exception as e:
        return None, str(e)

def main():
    st.markdown("<h1 style='text-align:center;color:#e94560;'>🎯 Fraud Prediction & Explanation</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not st.session_state.get('trained_models'):
        st.warning("⚠️ No trained models! Go to Model Training page first.")
        return
    
    model_name = st.session_state.get('selected_model', list(st.session_state.trained_models.keys())[0])
    model = st.session_state.trained_models[model_name]
    scaler = st.session_state.scaler
    feature_cols = st.session_state.feature_cols
    
    # Get training data for SHAP background if available
    X_train = st.session_state.get('X_train', None)
    
    st.info(f"🤖 Using model: **{model_name}**")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📝 Transaction Details")
        amount = st.number_input("Transaction Amount ($)", 0.0, 100000.0, 150.0)
        user_avg = st.number_input("User Avg Amount ($)", 0.0, 50000.0, 100.0)
        time_diff = st.number_input("Time Since Last (min)", 0.0, 1000.0, 30.0)
        transactions_5min = st.number_input("Txns Last 5 Min", 0, 50, 0)
        new_country = st.toggle("New Country")
        is_night = st.toggle("Night Transaction (0-6 AM)")
        is_weekend = st.toggle("Weekend Transaction")
        user_txn_count = st.number_input("User Total Txns", 1, 10000, 50)
        
        if st.button("🔍 Predict & Explain", use_container_width=True, type="primary"):
            # Predict
            amount_deviation = (amount - user_avg) / max(user_avg * 0.5, 1)
            
            feature_map = {
                'transaction_time_diff': time_diff,
                'amount_deviation': amount_deviation,
                'transactions_last_5min': transactions_5min,
                'new_country_flag': int(new_country),
                'night_transaction_flag': int(is_night),
                'is_weekend': int(is_weekend),
                'user_transaction_count': user_txn_count,
                'amount': amount
            }
            
            features = [feature_map.get(col, 0) for col in feature_cols]
            prediction, proba, risk, color, X_scaled = predict_fraud(features, model, scaler)
            
            # Show Prediction
            st.markdown("---")
            st.markdown(f"""
            <div style='background:linear-gradient(145deg,#16213e,#1a1a2e);padding:1rem;border-radius:10px;text-align:center;border:2px solid {color};'>
                <h3 style='color:{color};margin:0;'>{risk} ({proba*100:.1f}%)</h3>
            </div>
            """, unsafe_allow_html=True)
            
            triggered = check_rules(amount, user_avg, time_diff, transactions_5min, new_country, is_night)
            if triggered:
                st.warning(f"⚠️ {len(triggered)} Rules Triggered")
            
            # SHAP Explanation
            if 'shap' in st.session_state:
                del st.session_state.shap # Clear previous
            
            with st.spinner("Generating SHAP explanation..."):
                shap_obj, msg = plot_shap_waterfall(model, X_scaled, feature_cols, X_train)
                
                if shap_obj is not None:
                    st.session_state.shap_obj = shap_obj
                else:
                    st.error(f"Could not generate SHAP plot: {msg}")

    with col2:
        if 'shap_obj' in st.session_state:
            st.markdown("### 🧠 AI Explanation (SHAP)")
            st.markdown("This chart shows **exactly** how each feature pushed the fraud probability up (Red) or down (Blue) from the baseline.")
            
            try:
                # Create figure
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(st.session_state.shap_obj, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Error displaying plot: {str(e)}. Try selecting a Tree-based model.")
                
            st.markdown("""
            #### How to read this chart:
            - **Baseline (E[f(x)]):** The average fraud probability across all data.
            - **Red Bars:** Features that **increase** fraud risk.
            - **Blue Bars:** Features that **decrease** fraud risk.
            - **Final Value (f(x)):** The predicted score for *this* transaction.
            """)
        else:
            st.info("👈 Enter transaction details and click 'Predict' to see the AI explanation.")
            
            # Placeholder image or text
            st.markdown("""
            <div style='text-align:center;padding:3rem;opacity:0.5;'>
                <h1>🧠</h1>
                <p>AI Explanation Area</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
