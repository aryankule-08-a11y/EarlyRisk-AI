"""
📏 Weak Supervision Rules Page
Create rule-based fraud signals using domain knowledge
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="📏 Weak Supervision Rules", page_icon="📏", layout="wide")

def apply_weak_supervision_rules(df):
    """Apply weak supervision labeling functions"""
    df = df.copy()
    amount_col = next((c for c in ['amount', 'transaction_amount'] if c in df.columns), None)
    
    # Rule 1: Amount > 3x user average
    if amount_col and 'user_avg_amount' in df.columns:
        df['rule_amount'] = (df[amount_col] > df['user_avg_amount'] * 3).astype(int)
    else:
        df['rule_amount'] = 0
    
    # Rule 2: Transaction time diff < 2 minutes
    if 'transaction_time_diff' in df.columns:
        df['rule_time'] = ((df['transaction_time_diff'] < 2) & (df['transaction_time_diff'] != 999)).astype(int)
    else:
        df['rule_time'] = 0
    
    # Rule 3: New country flag
    if 'new_country_flag' in df.columns:
        df['rule_country'] = df['new_country_flag'].astype(int)
    else:
        df['rule_country'] = 0
    
    # Rule 4: High velocity (>=3 transactions in 5 min)
    if 'transactions_last_5min' in df.columns:
        df['rule_velocity'] = (df['transactions_last_5min'] >= 3).astype(int)
    else:
        df['rule_velocity'] = 0
    
    # Rule 5: Night transaction
    if 'night_transaction_flag' in df.columns:
        df['rule_night'] = df['night_transaction_flag'].astype(int)
    else:
        df['rule_night'] = 0
    
    # Calculate rule sum
    rule_cols = ['rule_amount', 'rule_time', 'rule_country', 'rule_velocity']
    df['rule_sum'] = df[rule_cols].sum(axis=1)
    
    # Create weak label: Fraud if sum >= 2
    df['weak_label'] = (df['rule_sum'] >= 2).astype(int)
    
    return df

def main():
    st.markdown("<h1 style='text-align:center;color:#e94560;'>📏 Weak Supervision Rules</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#a8a8a8;'>Generate fraud labels using rule-based labeling functions</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.get('feature_engineered_data') is None:
        st.warning("⚠️ Features not engineered yet! Go to Feature Engineering page first.")
        return
    
    df = st.session_state.feature_engineered_data
    
    # Rule Explanations
    st.markdown("### 📋 Labeling Functions (Rules)")
    
    cols = st.columns(2)
    with cols[0]:
        st.markdown("""
        <div style='background:#16213e;padding:1rem;border-radius:10px;border-left:4px solid #e94560;margin:0.5rem 0;'>
            <h4 style='color:#e94560;'>Rule 1: High Amount 💰</h4>
            <code>amount > user_average * 3</code>
            <p style='color:#a8a8a8;'>Flags transactions 3x higher than user's average</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background:#16213e;padding:1rem;border-radius:10px;border-left:4px solid #533483;margin:0.5rem 0;'>
            <h4 style='color:#533483;'>Rule 2: Rapid Transaction ⚡</h4>
            <code>transaction_time_diff < 2 minutes</code>
            <p style='color:#a8a8a8;'>Flags transactions happening too quickly</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div style='background:#16213e;padding:1rem;border-radius:10px;border-left:4px solid #4ade80;margin:0.5rem 0;'>
            <h4 style='color:#4ade80;'>Rule 3: New Country 🌍</h4>
            <code>new_country_flag == 1</code>
            <p style='color:#a8a8a8;'>Flags transactions from new locations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background:#16213e;padding:1rem;border-radius:10px;border-left:4px solid #60a5fa;margin:0.5rem 0;'>
            <h4 style='color:#60a5fa;'>Rule 4: High Velocity 🔄</h4>
            <code>transactions_last_5min >= 3</code>
            <p style='color:#a8a8a8;'>Flags burst of transactions in short time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🏷️ Weak Label Generation")
    st.markdown("**Formula:** `weak_label = 1 if (rule_sum >= 2) else 0`")
    
    if st.button("🏷️ Apply Weak Supervision Rules", use_container_width=True):
        with st.spinner("Applying rules..."):
            labeled_df = apply_weak_supervision_rules(df)
            st.session_state.weak_labeled_data = labeled_df
            st.success("✅ Weak labels generated!")
    
    if st.session_state.get('weak_labeled_data') is not None:
        df = st.session_state.weak_labeled_data
        
        st.markdown("---")
        st.markdown("### 📊 Rule Activation Results")
        
        # Statistics
        rule_cols = ['rule_amount', 'rule_time', 'rule_country', 'rule_velocity']
        
        cols = st.columns(5)
        for i, rule in enumerate(rule_cols):
            if rule in df.columns:
                count = df[rule].sum()
                pct = count / len(df) * 100
                with cols[i]:
                    st.metric(rule.replace('rule_', '').title(), f"{int(count):,}", f"{pct:.1f}%")
        
        with cols[4]:
            fraud_count = df['weak_label'].sum()
            fraud_pct = fraud_count / len(df) * 100
            st.metric("Fraud Labels", f"{int(fraud_count):,}", f"{fraud_pct:.1f}%")
        
        # Visualizations
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 Rule Distribution", "🏷️ Label Distribution", "📋 Data Preview"])
        
        with tab1:
            rule_counts = {rule.replace('rule_', '').title(): df[rule].sum() for rule in rule_cols if rule in df.columns}
            fig = px.bar(x=list(rule_counts.keys()), y=list(rule_counts.values()),
                        color=list(rule_counts.values()), color_continuous_scale='Reds',
                        labels={'x': 'Rule', 'y': 'Activations'}, title='Rule Activation Frequency')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            label_counts = df['weak_label'].value_counts()
            fig = px.pie(values=label_counts.values, names=['Non-Fraud', 'Fraud'],
                        color_discrete_sequence=['#4ade80', '#e94560'], title='Weak Label Distribution')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            display_cols = rule_cols + ['rule_sum', 'weak_label']
            existing = [c for c in display_cols if c in df.columns]
            st.dataframe(df[existing].head(20), use_container_width=True)
        
        st.info("👉 Navigate to **Model Training** page to train ML models on weak labels.")

if __name__ == "__main__":
    main()
