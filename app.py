"""
🚨 Early Fraud Signal Detection using Weak Supervision
Main Application Entry Point

This is a production-ready Streamlit application that demonstrates
early fraud detection using weak supervision techniques.

Author: Aryan Kule
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🚨 Early Fraud Signal Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/aryankule-08-a11y',
        'Report a bug': 'https://github.com/aryankule-08-a11y',
        'About': '# 🚨 Early Fraud Signal Detection\nBuilt with ❤️ using Weak Supervision'
    }
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main Container Styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid #e94560;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e94560;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #e94560 0%, #533483 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        color: #a8a8a8;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card Styling */
    .metric-card {
        background: linear-gradient(145deg, #1e1e30 0%, #2d2d44 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #e94560;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(233, 69, 96, 0.3);
    }
    
    .feature-card {
        background: linear-gradient(145deg, #16213e 0%, #1a1a2e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #e94560;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: linear-gradient(145deg, #1f2b47 0%, #242438 100%);
        border-left: 4px solid #533483;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #e94560 0%, #533483 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.5);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e94560 0%, #533483 100%);
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Alert Boxes */
    .success-box {
        background: linear-gradient(145deg, #1e4d2b 0%, #2d6e3f 100%);
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #4ade80;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(145deg, #4d4d1e 0%, #6e6e2d 100%);
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #fbbf24;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: linear-gradient(145deg, #4d1e1e 0%, #6e2d2d 100%);
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #ef4444;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(145deg, #1e3a4d 0%, #2d5a6e 100%);
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #60a5fa;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #a8a8a8;
        border-top: 1px solid #333;
        margin-top: 3rem;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 10px 20px;
        border: 1px solid #333;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #e94560 0%, #533483 100%);
        border: none;
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        color: #e94560;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a8a8a8;
    }
    
    /* Input Fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a2e;
        border-radius: 8px;
    }
    
    /* Animation Keyframes */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .pulsing {
        animation: pulse 2s infinite;
    }
    
    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(90deg, #e94560, #533483, #e94560);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s linear infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'feature_engineered_data' not in st.session_state:
        st.session_state.feature_engineered_data = None
    if 'weak_labeled_data' not in st.session_state:
        st.session_state.weak_labeled_data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None

init_session_state()

# =============================================================================
# HOME PAGE CONTENT
# =============================================================================

def main():
    """Main function for the Home page"""
    
    # Header Section
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="main-header">🚨 Early Fraud Signal Detection</h1>
            <h2 class="sub-header">Using Weak Supervision Techniques</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Introduction Section
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What is This Project?
        
        This application demonstrates **Early Fraud Signal Detection** using **Weak Supervision** - 
        a cutting-edge machine learning technique that allows us to detect fraudulent transactions 
        without requiring extensive manually labeled data.
        
        Instead of relying on expensive human annotation, we use **programmatic labeling functions** 
        (rules) to generate weak labels, which are then used to train machine learning models.
        """)
        
        st.markdown("""
        ### 💡 Key Concepts
        
        <div class="feature-card">
            <h4>🏷️ Weak Supervision</h4>
            <p>A paradigm where noisy, limited, or imprecise sources are used to create training labels 
            programmatically, reducing the need for manual labeling.</p>
        </div>
        
        <div class="feature-card">
            <h4>⚡ Early Detection</h4>
            <p>Detecting fraud signals as early as possible - sometimes even from the first transaction - 
            rather than waiting for patterns to emerge over time.</p>
        </div>
        
        <div class="feature-card">
            <h4>📊 Rule-Based Signals</h4>
            <p>Using domain knowledge to create rules (labeling functions) that capture suspicious 
            behavior patterns in transaction data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### 📈 Quick Stats
        """)
        
        st.metric(label="Detection Speed", value="< 1 sec", delta="Real-time")
        st.metric(label="Rule Coverage", value="4 Rules", delta="Expandable")
        st.metric(label="Model Types", value="3 Models", delta="Ensemble Ready")
        
        st.markdown("---")
        
        st.markdown("""
        ### 🔧 Features
        - ✅ CSV Data Upload
        - ✅ Automatic Feature Engineering
        - ✅ Weak Supervision Rules
        - ✅ Multiple ML Models
        - ✅ Real-time Prediction
        - ✅ Visual Analytics
        - ✅ Early Detection Simulation
        """)
    
    st.markdown("---")
    
    # Problem Statement Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔴 The Problem
        
        Traditional fraud detection systems face several challenges:
        
        1. **Delayed Detection**: Fraud is often detected only after significant damage
        2. **Limited Labeled Data**: Getting labeled fraud data is expensive and time-consuming
        3. **Evolving Patterns**: Fraudsters constantly change their tactics
        4. **Class Imbalance**: Fraud cases are typically rare (< 1% of transactions)
        5. **Real-time Requirements**: Decisions must be made in milliseconds
        """)
        
        st.error("💸 Global fraud losses exceed $30 billion annually!")
    
    with col2:
        st.markdown("""
        ### 🟢 Our Solution
        
        This application addresses these challenges through:
        
        1. **Early Signals**: Detect suspicious patterns from the first transaction
        2. **Weak Supervision**: Generate labels without manual annotation
        3. **Adaptive Rules**: Easily update rules as fraud patterns evolve
        4. **Ensemble Models**: Combine multiple models for robust detection
        5. **Explainable AI**: Understand why each prediction was made
        """)
        
        st.success("🎯 Detect fraud up to 10x faster with early signals!")
    
    st.markdown("---")
    
    # How It Works Section
    st.markdown("""
    ### 🔄 How It Works
    """)
    
    workflow_cols = st.columns(5)
    
    with workflow_cols[0]:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h2>📤</h2>
            <h4>1. Upload</h4>
            <p>Upload your transaction CSV data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[1]:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h2>⚙️</h2>
            <h4>2. Engineer</h4>
            <p>Automatic feature engineering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[2]:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h2>📏</h2>
            <h4>3. Label</h4>
            <p>Apply weak supervision rules</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[3]:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h2>🤖</h2>
            <h4>4. Train</h4>
            <p>Train ML models on weak labels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[4]:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h2>🎯</h2>
            <h4>5. Predict</h4>
            <p>Detect fraud in real-time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Getting Started Section
    st.markdown("""
    ### 🚀 Getting Started
    
    Use the **sidebar navigation** to explore each section of the application:
    
    | Page | Description |
    |------|-------------|
    | 📊 **Data Upload & Preview** | Upload and explore your transaction data |
    | 🔧 **Feature Engineering** | Create powerful features from raw data |
    | 📏 **Weak Supervision Rules** | Define and apply fraud detection rules |
    | 🤖 **Model Training** | Train and evaluate ML models |
    | 🎯 **Fraud Prediction** | Make real-time fraud predictions |
    | ⏰ **Early Detection** | Analyze early fraud signals |
    | 📈 **Visualizations** | Explore interactive charts and graphs |
    | 📝 **Conclusion** | Summary and future directions |
    """)
    
    st.markdown("---")
    
    # Sample Data Section
    st.markdown("""
    ### 📋 Don't Have Data?
    
    No worries! You can generate sample transaction data to explore the application.
    """)
    
    if st.button("🎲 Generate Sample Data", use_container_width=True):
        sample_data = generate_sample_data()
        st.session_state.data = sample_data
        st.success("✅ Sample data generated successfully! Navigate to 'Data Upload & Preview' to explore.")
        st.dataframe(sample_data.head(10))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🚨 <strong>Early Fraud Signal Detection using Weak Supervision</strong></p>
        <p>Built with ❤️ by <strong>Aryan Kule</strong></p>
        <p>Powered by Streamlit | Python | Scikit-learn</p>
        <p>© 2026 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)


def generate_sample_data(n_samples=1000):
    """Generate sample transaction data for demonstration"""
    np.random.seed(42)
    
    # Generate user IDs
    user_ids = [f"USER_{i:04d}" for i in np.random.randint(1, 201, n_samples)]
    
    # Generate timestamps over the last 30 days
    base_date = datetime(2026, 1, 1)
    timestamps = pd.date_range(start=base_date, periods=n_samples, freq='T')
    timestamps = timestamps + pd.to_timedelta(np.random.randint(0, 30*24*60, n_samples), unit='m')
    timestamps = sorted(timestamps)
    
    # Generate amounts (normal transactions + some suspicious ones)
    amounts = np.random.exponential(100, n_samples)
    # Add some suspicious high amounts
    suspicious_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    amounts[suspicious_indices] *= np.random.uniform(5, 15, len(suspicious_indices))
    amounts = np.round(amounts, 2)
    
    # Generate countries
    countries = np.random.choice(
        ['USA', 'UK', 'Canada', 'Germany', 'France', 'India', 'Japan', 'Brazil', 'Unknown'],
        n_samples,
        p=[0.3, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
    )
    
    # Generate merchants
    merchants = np.random.choice(
        ['Amazon', 'Walmart', 'Target', 'BestBuy', 'Apple', 'Netflix', 'Spotify', 
         'Uber', 'DoorDash', 'Unknown_Merchant'],
        n_samples,
        p=[0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1]
    )
    
    # Generate transaction types
    transaction_types = np.random.choice(
        ['purchase', 'transfer', 'withdrawal', 'refund', 'payment'],
        n_samples,
        p=[0.4, 0.2, 0.15, 0.1, 0.15]
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'timestamp': timestamps,
        'amount': amounts,
        'country': countries,
        'merchant': merchants,
        'transaction_type': transaction_types
    })
    
    return df


if __name__ == "__main__":
    main()
