# 🚨 Early Fraud Signal Detection using Weak Supervision

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://earlyrisk-ai.streamlit.app/)

A production-ready Streamlit web application that demonstrates **Early Fraud Detection** using **Weak Supervision** techniques. This project shows how to detect fraudulent transactions without expensive manual labeling by using rule-based labeling functions.

## 🔴 Live Demo
👉 **[Click here to view the live app](https://earlyrisk-ai.streamlit.app/)**

## 🎯 Features

- 📤 **Data Upload & Preview** - Upload CSV transaction data with automatic column detection
- 🔧 **Feature Engineering** - Automatic creation of fraud-detection features
- 📏 **Weak Supervision Rules** - Rule-based labeling functions for fraud detection
- 🤖 **Model Training** - Train Logistic Regression, Random Forest, and XGBoost models
- 🎯 **Real-time Prediction** - Make fraud predictions on new transactions
- ⏰ **Early Detection Analysis** - Compare early vs traditional detection methods
- 📈 **Interactive Visualizations** - Comprehensive charts and analytics
- 📝 **Conclusion & Insights** - Summary with real-world applications

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/aryankule-08-a11y/EarlyRisk-AI.git
cd Fraud_Signal_Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser at `http://localhost:8501`

## 📊 Project Structure

```
Fraud_Signal_Detection/
├── app.py                          # Main application (Home page)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── pages/
    ├── 1_📊_Data_Upload.py        # Data upload and preview
    ├── 2_🔧_Feature_Engineering.py # Feature creation
    ├── 3_📏_Weak_Supervision.py   # Rule-based labeling
    ├── 4_🤖_Model_Training.py     # ML model training
    ├── 5_🎯_Fraud_Prediction.py   # Real-time prediction
    ├── 6_⏰_Early_Detection.py    # Detection analysis
    ├── 7_📈_Visualizations.py     # Charts and graphs
    └── 8_📝_Conclusion.py         # Summary and insights
```

## 🏷️ Weak Supervision Concept

Instead of expensive manual labeling, we use **programmatic labeling functions** (rules) to generate weak labels:

| Rule | Description | Formula |
|------|-------------|---------|
| **High Amount** | Transaction > 3x user average | `amount > user_avg * 3` |
| **Rapid Transaction** | Time since last < 2 minutes | `time_diff < 2` |
| **New Country** | First transaction from country | `new_country_flag == 1` |
| **High Velocity** | 3+ transactions in 5 minutes | `transactions_5min >= 3` |

**Weak Label Formula:** `fraud = 1 if (rule_sum >= 2) else 0`

## 📈 Expected Data Format

The application expects CSV data with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Unique user identifier |
| `timestamp` | datetime | Transaction timestamp |
| `amount` | float | Transaction amount |
| `country` | string | Transaction country |
| `merchant` | string | Merchant name |
| `transaction_type` | string | Type of transaction |

**Note:** Missing columns are handled gracefully. You can also generate sample data within the app.

## 🛠️ Tech Stack

- **Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Plotly, Matplotlib, Seaborn

## 📷 Screenshots

### Home Page
Modern, clean interface with project overview and getting started guide.

### Feature Engineering
Automatic creation of fraud detection features with explanations.

### Model Training
Train multiple ML models and compare performance metrics.

### Fraud Prediction
Real-time predictions with rule explanations.

## 🔮 Future Improvements

- 🧠 Deep Learning (Graph Neural Networks, LSTM)
- 📊 Concept Drift Detection
- ⚡ Real-time Streaming (Apache Kafka)
- 🔍 SHAP Explainability

## 👤 Author

**Aryan Kule**

- GitHub: [@aryankule-08-a11y](https://github.com/aryankule-08-a11y)

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Snorkel AI](https://snorkel.ai/) for weak supervision concepts
- [Streamlit](https://streamlit.io/) for the amazing framework
- [Scikit-learn](https://scikit-learn.org/) for ML tools

---

<p align="center">
  Made with ❤️ for the Data Science Community
</p>
