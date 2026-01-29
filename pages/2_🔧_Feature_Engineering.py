"""
🔧 Feature Engineering Page
Creates advanced features for fraud detection
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="🔧 Feature Engineering", page_icon="🔧", layout="wide")

@st.cache_data
def engineer_features(df):
    """Create advanced features for fraud detection"""
    df = df.copy()
    
    # Find columns
    timestamp_col = next((c for c in ['timestamp', 'date', 'datetime'] if c in df.columns), None)
    user_col = next((c for c in ['user_id', 'customer_id', 'account_id'] if c in df.columns), None)
    amount_col = next((c for c in ['amount', 'transaction_amount', 'value'] if c in df.columns), None)
    country_col = next((c for c in ['country', 'location', 'geo'] if c in df.columns), None)
    
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        df = df.sort_values([timestamp_col])
    
    # Feature 1: Transaction Time Difference
    if timestamp_col and user_col:
        df['transaction_time_diff'] = df.groupby(user_col)[timestamp_col].diff().dt.total_seconds() / 60
        df['transaction_time_diff'] = df['transaction_time_diff'].fillna(999)
    else:
        df['transaction_time_diff'] = 999
    
    # Feature 2: Amount Deviation
    if amount_col and user_col:
        user_avg = df.groupby(user_col)[amount_col].transform('mean')
        user_std = df.groupby(user_col)[amount_col].transform('std').fillna(1)
        df['user_avg_amount'] = user_avg
        df['amount_deviation'] = (df[amount_col] - user_avg) / user_std.replace(0, 1)
    else:
        df['user_avg_amount'] = 0
        df['amount_deviation'] = 0
    
    # Feature 3: Transactions Last 5 Minutes
    df['transactions_last_5min'] = 0
    if timestamp_col and user_col:
        df = df.sort_values(timestamp_col)
        for user in df[user_col].unique():
            mask = df[user_col] == user
            user_df = df[mask]
            counts = []
            for idx, row in user_df.iterrows():
                t = row[timestamp_col]
                cnt = ((user_df[timestamp_col] >= t - timedelta(minutes=5)) & (user_df[timestamp_col] < t)).sum()
                counts.append(cnt)
            df.loc[mask, 'transactions_last_5min'] = counts
    
    # Feature 4: New Country Flag
    df['new_country_flag'] = 0
    if country_col and user_col:
        for user in df[user_col].unique():
            mask = df[user_col] == user
            user_df = df[mask]
            seen = set()
            flags = []
            for _, row in user_df.iterrows():
                c = row[country_col]
                flags.append(1 if c not in seen and len(seen) > 0 else 0)
                seen.add(c)
            df.loc[mask, 'new_country_flag'] = flags
    
    # Feature 5: Night Transaction Flag
    if timestamp_col:
        df['transaction_hour'] = df[timestamp_col].dt.hour
        df['night_transaction_flag'] = ((df['transaction_hour'] >= 0) & (df['transaction_hour'] < 6)).astype(int)
    else:
        df['transaction_hour'] = 12
        df['night_transaction_flag'] = 0
    
    # Feature 6: Weekend Flag
    if timestamp_col:
        df['is_weekend'] = (df[timestamp_col].dt.dayofweek >= 5).astype(int)
    else:
        df['is_weekend'] = 0
    
    # Feature 7: User Transaction Count
    if user_col:
        df['user_transaction_count'] = df.groupby(user_col).cumcount() + 1
    else:
        df['user_transaction_count'] = 1
    
    return df

def main():
    st.markdown("<h1 style='text-align:center;color:#e94560;'>🔧 Feature Engineering</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.get('data') is None:
        st.warning("⚠️ No data found! Please upload data first.")
        return
    
    df = st.session_state.data
    
    st.markdown("### 📚 Features to Create")
    cols = st.columns(2)
    with cols[0]:
        st.info("⏱️ **Transaction Time Diff** - Minutes since last transaction")
        st.info("📊 **Amount Deviation** - Z-score from user's average")
        st.info("🔄 **Transactions Last 5 Min** - Velocity check")
    with cols[1]:
        st.info("🌍 **New Country Flag** - Transaction from new location")
        st.info("🌙 **Night Transaction Flag** - 12 AM - 6 AM")
        st.info("📅 **Weekend Flag** - Saturday/Sunday")
    
    if st.button("🚀 Engineer Features", use_container_width=True):
        with st.spinner("Engineering features..."):
            engineered_df = engineer_features(df)
            st.session_state.feature_engineered_data = engineered_df
            st.session_state.processed_data = engineered_df
            st.success("✅ Features engineered!")
    
    if st.session_state.get('feature_engineered_data') is not None:
        fe_df = st.session_state.feature_engineered_data
        st.markdown("### 📋 Feature Preview")
        new_cols = ['transaction_time_diff', 'amount_deviation', 'transactions_last_5min', 'new_country_flag', 'night_transaction_flag']
        existing = [c for c in new_cols if c in fe_df.columns]
        st.dataframe(fe_df[existing].head(15), use_container_width=True)
        st.info("👉 Navigate to **Weak Supervision Rules** page next.")

if __name__ == "__main__":
    main()
