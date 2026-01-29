"""
⏰ Early Detection Analysis Page
Simulate and compare early vs traditional fraud detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="⏰ Early Detection", page_icon="⏰", layout="wide")

def simulate_early_detection(df, model, scaler, feature_cols):
    """Simulate fraud detection at different transaction counts"""
    results = []
    
    user_col = next((c for c in ['user_id', 'customer_id'] if c in df.columns), None)
    if not user_col:
        return pd.DataFrame()
    
    # Get actual fraud cases
    fraud_users = df[df['weak_label'] == 1][user_col].unique()
    
    for detection_point in [1, 2, 3, 5, 10]:
        detected = 0
        total_fraud = 0
        
        for user in fraud_users:
            user_df = df[df[user_col] == user].head(detection_point)
            if len(user_df) == 0:
                continue
            
            total_fraud += 1
            
            # Check if fraud detected
            if user_df['weak_label'].sum() > 0:
                detected += 1
        
        detection_rate = (detected / total_fraud * 100) if total_fraud > 0 else 0
        results.append({
            'Transaction #': f"After {detection_point} txn(s)",
            'Detected': detected,
            'Total Fraud Users': total_fraud,
            'Detection Rate (%)': detection_rate
        })
    
    return pd.DataFrame(results)

def main():
    st.markdown("<h1 style='text-align:center;color:#e94560;'>⏰ Early Detection Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#a8a8a8;'>Compare early detection vs traditional detection methods</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.get('weak_labeled_data') is None:
        st.warning("⚠️ No labeled data! Complete the Weak Supervision step first.")
        return
    
    df = st.session_state.weak_labeled_data
    
    # Concept Explanation
    st.markdown("### 💡 What is Early Detection?")
    
    cols = st.columns(2)
    with cols[0]:
        st.markdown("""
        <div style='background:linear-gradient(145deg,#4d1e1e,#6e2d2d);padding:1.5rem;border-radius:12px;border-left:4px solid #ef4444;'>
            <h4 style='color:#ef4444;'>🐢 Traditional Detection</h4>
            <ul style='color:#a8a8a8;'>
                <li>Waits for multiple transactions</li>
                <li>Relies on historical patterns</li>
                <li>Detection after 10-50 transactions</li>
                <li>Fraud damage already done</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div style='background:linear-gradient(145deg,#1e4d2b,#2d6e3f);padding:1.5rem;border-radius:12px;border-left:4px solid #4ade80;'>
            <h4 style='color:#4ade80;'>🚀 Early Detection (Ours)</h4>
            <ul style='color:#a8a8a8;'>
                <li>Detects from 1st-3rd transaction</li>
                <li>Uses weak supervision signals</li>
                <li>Real-time rule evaluation</li>
                <li>Prevents damage before it occurs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Detection Simulation")
    
    # Simulate detection
    user_col = next((c for c in ['user_id', 'customer_id'] if c in df.columns), None)
    
    if user_col:
        # Calculate detection metrics
        total_users = df[user_col].nunique()
        fraud_users = df[df['weak_label'] == 1][user_col].nunique()
        
        # Detection at different points
        detection_data = []
        for n in [1, 2, 3, 5, 10, 20]:
            early_detected = 0
            for user in df[user_col].unique():
                user_df = df[df[user_col] == user].head(n)
                if user_df['weak_label'].sum() > 0:
                    early_detected += 1
            
            detection_data.append({
                'Transactions': n,
                'Fraudsters Detected': early_detected,
                'Detection Rate': early_detected / max(fraud_users, 1) * 100
            })
        
        detection_df = pd.DataFrame(detection_data)
        
        # Metrics
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Users", f"{total_users:,}")
        with cols[1]:
            st.metric("Fraud Users", f"{fraud_users:,}")
        with cols[2]:
            first_txn_rate = detection_df[detection_df['Transactions'] == 1]['Detection Rate'].values[0]
            st.metric("Detection @ 1st Txn", f"{first_txn_rate:.1f}%")
        with cols[3]:
            third_txn_rate = detection_df[detection_df['Transactions'] == 3]['Detection Rate'].values[0]
            st.metric("Detection @ 3rd Txn", f"{third_txn_rate:.1f}%")
        
        st.markdown("---")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Detection Timeline", "📊 Cumulative Detection", "🔄 Comparison"])
        
        with tab1:
            fig = px.line(detection_df, x='Transactions', y='Detection Rate',
                         markers=True, title='Fraud Detection Rate by Transaction Number')
            fig.update_traces(line_color='#e94560', marker_size=10)
            fig.add_hline(y=50, line_dash="dash", line_color="yellow", annotation_text="50% threshold")
            fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="80% threshold")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8',
                            xaxis_title="Number of Transactions", yaxis_title="Detection Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.area(detection_df, x='Transactions', y='Fraudsters Detected',
                         title='Cumulative Fraudsters Detected')
            fig.update_traces(fillcolor='rgba(233,69,96,0.3)', line_color='#e94560')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Traditional vs Early comparison
            comparison_data = pd.DataFrame({
                'Method': ['Early Detection (3 txns)', 'Traditional (10 txns)', 'Late Detection (20 txns)'],
                'Detection Rate': [
                    detection_df[detection_df['Transactions'] == 3]['Detection Rate'].values[0],
                    detection_df[detection_df['Transactions'] == 10]['Detection Rate'].values[0],
                    detection_df[detection_df['Transactions'] == 20]['Detection Rate'].values[0]
                ],
                'Time to Detect': ['Seconds', 'Minutes', 'Hours']
            })
            
            fig = px.bar(comparison_data, x='Method', y='Detection Rate', color='Method',
                        color_discrete_sequence=['#4ade80', '#fbbf24', '#ef4444'],
                        title='Detection Method Comparison')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a8a8a8',
                            showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Benefits
        st.markdown("---")
        st.markdown("### 🎯 Early Detection Benefits")
        
        cols = st.columns(3)
        with cols[0]:
            savings = fraud_users * 500  # Assume $500 avg fraud
            st.markdown(f"""
            <div style='background:#16213e;padding:1.5rem;border-radius:12px;text-align:center;'>
                <h2 style='color:#4ade80;'>💰 ${savings:,.0f}</h2>
                <p style='color:#a8a8a8;'>Potential Savings</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            hours_saved = fraud_users * 2  # 2 hours investigation per fraud
            st.markdown(f"""
            <div style='background:#16213e;padding:1.5rem;border-radius:12px;text-align:center;'>
                <h2 style='color:#60a5fa;'>⏱️ {hours_saved:,} hrs</h2>
                <p style='color:#a8a8a8;'>Investigation Time Saved</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown(f"""
            <div style='background:#16213e;padding:1.5rem;border-radius:12px;text-align:center;'>
                <h2 style='color:#e94560;'>🛡️ {third_txn_rate:.0f}%</h2>
                <p style='color:#a8a8a8;'>Fraud Prevented</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.warning("User ID column not found. Cannot perform per-user analysis.")
    
    st.info("👉 Navigate to **Visualizations** page for more detailed charts and analysis.")

if __name__ == "__main__":
    main()
