"""
📊 Data Upload & Preview Page
Handles CSV file upload, data preview, and basic statistics
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="📊 Data Upload | Fraud Detection",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .upload-box {
        background: linear-gradient(145deg, #1e1e30 0%, #2d2d44 100%);
        border: 2px dashed #e94560;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #533483;
        box-shadow: 0 8px 25px rgba(233, 69, 96, 0.2);
    }
    
    .stat-card {
        background: linear-gradient(145deg, #16213e 0%, #1a1a2e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e94560;
    }
    
    .stat-label {
        color: #a8a8a8;
        font-size: 0.9rem;
    }
    
    .column-type-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.25rem;
    }
    
    .numeric-badge {
        background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%);
        color: white;
    }
    
    .categorical-badge {
        background: linear-gradient(90deg, #60a5fa 0%, #3b82f6 100%);
        color: white;
    }
    
    .datetime-badge {
        background: linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%);
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_column_types(df):
    """Automatically detect column types"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to detect datetime columns that are stored as strings
    for col in categorical_cols[:]:
        try:
            pd.to_datetime(df[col], errors='raise')
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except:
            pass
    
    return numeric_cols, categorical_cols, datetime_cols


def get_missing_value_summary(df):
    """Get missing value statistics"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })
    return missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)


def generate_sample_data(n_samples=1000):
    """Generate sample transaction data for demonstration"""
    np.random.seed(42)
    
    user_ids = [f"USER_{i:04d}" for i in np.random.randint(1, 201, n_samples)]
    
    base_date = datetime(2026, 1, 1)
    timestamps = pd.date_range(start=base_date, periods=n_samples, freq='T')
    timestamps = timestamps + pd.to_timedelta(np.random.randint(0, 30*24*60, n_samples), unit='m')
    timestamps = sorted(timestamps)
    
    amounts = np.random.exponential(100, n_samples)
    suspicious_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    amounts[suspicious_indices] *= np.random.uniform(5, 15, len(suspicious_indices))
    amounts = np.round(amounts, 2)
    
    countries = np.random.choice(
        ['USA', 'UK', 'Canada', 'Germany', 'France', 'India', 'Japan', 'Brazil', 'Unknown'],
        n_samples,
        p=[0.3, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
    )
    
    merchants = np.random.choice(
        ['Amazon', 'Walmart', 'Target', 'BestBuy', 'Apple', 'Netflix', 'Spotify', 
         'Uber', 'DoorDash', 'Unknown_Merchant'],
        n_samples,
        p=[0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1]
    )
    
    transaction_types = np.random.choice(
        ['purchase', 'transfer', 'withdrawal', 'refund', 'payment'],
        n_samples,
        p=[0.4, 0.2, 0.15, 0.1, 0.15]
    )
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'timestamp': timestamps,
        'amount': amounts,
        'country': countries,
        'merchant': merchants,
        'transaction_type': transaction_types
    })
    
    return df


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    st.markdown("""
        <h1 style="text-align: center; color: #e94560;">📊 Data Upload & Preview</h1>
        <p style="text-align: center; color: #a8a8a8;">Upload your transaction data to begin fraud detection analysis</p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Upload Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file containing transaction data"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Try to parse timestamp column if it exists
                timestamp_cols = ['timestamp', 'date', 'datetime', 'time', 'created_at', 'transaction_date']
                for col in timestamp_cols:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except:
                            pass
                        break
                
                st.session_state.data = df
                st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.markdown("### 🎲 Generate Sample Data")
        st.markdown("Don't have data? Generate sample transaction data to explore the app.")
        
        n_samples = st.slider("Number of samples", 100, 5000, 1000, 100)
        
        if st.button("🎲 Generate Sample Data", use_container_width=True):
            sample_df = generate_sample_data(n_samples)
            st.session_state.data = sample_df
            st.success("✅ Sample data generated!")
    
    st.markdown("---")
    
    # Display Data if available
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Dataset Overview
        st.markdown("### 📋 Dataset Overview")
        
        # Stats Cards
        stat_cols = st.columns(5)
        
        with stat_cols[0]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{df.shape[0]:,}</div>
                <div class="stat-label">Total Rows</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_cols[1]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{df.shape[1]}</div>
                <div class="stat-label">Total Columns</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_cols[2]:
            missing_count = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{missing_count:,}</div>
                <div class="stat-label">Missing Values</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_cols[3]:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{memory_mb:.2f} MB</div>
                <div class="stat-label">Memory Usage</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_cols[4]:
            duplicates = df.duplicated().sum()
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{duplicates:,}</div>
                <div class="stat-label">Duplicates</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Column Type Detection
        st.markdown("### 🔍 Column Analysis")
        
        numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔢 Numeric Columns")
            if numeric_cols:
                for col in numeric_cols:
                    st.markdown(f'<span class="column-type-badge numeric-badge">{col}</span>', unsafe_allow_html=True)
            else:
                st.info("No numeric columns detected")
        
        with col2:
            st.markdown("#### 📝 Categorical Columns")
            if categorical_cols:
                for col in categorical_cols:
                    st.markdown(f'<span class="column-type-badge categorical-badge">{col}</span>', unsafe_allow_html=True)
            else:
                st.info("No categorical columns detected")
        
        with col3:
            st.markdown("#### 📅 DateTime Columns")
            if datetime_cols:
                for col in datetime_cols:
                    st.markdown(f'<span class="column-type-badge datetime-badge">{col}</span>', unsafe_allow_html=True)
            else:
                st.info("No datetime columns detected")
        
        st.markdown("---")
        
        # Data Preview Tabs
        st.markdown("### 📑 Data Preview")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 First 10 Rows", "📊 Data Types", "❓ Missing Values", "📈 Statistics"])
        
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str).values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with tab3:
            missing_df = get_missing_value_summary(df)
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
                
                # Missing value visualization
                st.markdown("#### Missing Value Distribution")
                import plotly.express as px
                fig = px.bar(
                    missing_df, 
                    x='Column', 
                    y='Missing %',
                    color='Missing %',
                    color_continuous_scale='Reds',
                    title='Missing Values by Column'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#a8a8a8'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values in the dataset!")
        
        with tab4:
            st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Required Columns Check
        st.markdown("### ✅ Required Columns Check")
        
        required_cols = ['user_id', 'amount', 'timestamp', 'country', 'merchant', 'transaction_type']
        
        check_results = []
        for col in required_cols:
            found = col in df.columns
            similar = None
            if not found:
                # Check for similar column names
                for df_col in df.columns:
                    if col.lower() in df_col.lower() or df_col.lower() in col.lower():
                        similar = df_col
                        break
            
            check_results.append({
                'Column': col,
                'Status': '✅ Found' if found else '⚠️ Missing',
                'Similar Column': similar if not found else '-'
            })
        
        check_df = pd.DataFrame(check_results)
        st.dataframe(check_df, use_container_width=True)
        
        missing_required = [r['Column'] for r in check_results if r['Status'] == '⚠️ Missing']
        if missing_required:
            st.warning(f"⚠️ Missing columns: {', '.join(missing_required)}. The app will handle these gracefully.")
        else:
            st.success("✅ All required columns are present! You can proceed to Feature Engineering.")
        
        # Column Mapping (if needed)
        if missing_required:
            st.markdown("### 🔄 Column Mapping (Optional)")
            st.markdown("Map your columns to the expected column names if they have different names:")
            
            mapping = {}
            for req_col in missing_required:
                available_cols = ['(Skip)'] + df.columns.tolist()
                selected = st.selectbox(
                    f"Map '{req_col}' to:",
                    available_cols,
                    key=f"map_{req_col}"
                )
                if selected != '(Skip)':
                    mapping[selected] = req_col
            
            if mapping and st.button("🔄 Apply Mapping"):
                df_mapped = df.rename(columns=mapping)
                st.session_state.data = df_mapped
                st.success("✅ Column mapping applied!")
                st.experimental_rerun()
        
        # Navigation
        st.markdown("---")
        st.markdown("### ➡️ Next Steps")
        st.info("👉 Navigate to **Feature Engineering** page to create fraud detection features from your data.")
        
    else:
        # No data uploaded yet
        st.markdown("""
        <div class="upload-box">
            <h2>📤 Upload Your Transaction Data</h2>
            <p>Drag and drop a CSV file or click to browse</p>
            <p style="color: #e94560;">Expected columns: user_id, amount, timestamp, country, merchant, transaction_type</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Tip:** You can also generate sample data using the button above to explore the application.")


if __name__ == "__main__":
    main()
