# 🔁 Customer Churn Prediction
### End-to-End Binary Classification · Logistic Regression · Random Forest · XGBoost · SHAP Explainability

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/profpius/customer-churn-prediction/blob/main/churn_prediction.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Key Results

- Built an end-to-end churn prediction system trained on **37,000 customer records**
- Achieved **94.02% F1 Score** and **97.60% ROC-AUC** using a tuned XGBoost pipeline
- Validated performance with **Stratified 5-Fold Cross-Validation**, confirming stability across all data splits
- Identified the top churn drivers using **SHAP explainability**, providing directional business insight beyond standard importance scores
- Delivered a **production-ready, serialised sklearn Pipeline** capable of scoring new customers in real time with a single function call
- Translated model outputs into actionable retention strategies with an estimated recovery potential of **₦150,000,000+ monthly revenue**

---

## 🧠 Skills Demonstrated

- End-to-End Machine Learning Pipeline Development
- Data Cleaning, Feature Engineering and Leakage Prevention
- Multi-Model Comparison and Hyperparameter Tuning
- Cross-Validation and Robust Performance Evaluation
- Model Explainability with SHAP (TreeExplainer)
- Production Pipeline Serialisation with `joblib`
- Business Insight Generation from Model Outputs

---

## 📌 Business Problem

Customer churn is one of the most costly challenges in subscription-based and retail businesses. Acquiring a new customer can cost **5 to 10× more** than retaining an existing one, yet most companies lack a systematic, data-driven early warning system.

This project addresses that gap by building a **production-ready machine learning pipeline** that:

- Identifies customers at high risk of churning **before** they leave
- Quantifies the exact business levers driving churn (loyalty points, membership tier, complaint history)
- Delivers a **serialised, deployment-ready pipeline** that scores new customers in real time with a single API call

The model enables retention teams to take **targeted, cost-efficient action** instead of blanketing the entire customer base with expensive campaigns.

---

## 🗂️ Project Overview

| Attribute | Detail |
|---|---|
| **Task** | Binary Classification (Churn = 1 / No Churn = 0) |
| **Dataset** | ~37,000 customer records, 24 raw features |
| **Best Model** | XGBoost |
| **F1 Score** | **94.02%** (Tuned XGBoost) |
| **ROC-AUC** | **97.60%** |
| **Explainability** | SHAP (TreeExplainer) |
| **Deployment** | `joblib` serialised sklearn Pipeline |
| **Validation** | Stratified 5-Fold Cross-Validation |

---

## 🔄 Machine Learning Pipeline

The model follows a structured, reproducible pipeline that prevents data leakage at every stage.

```
Raw Data
   |
Data Cleaning
   |
Train / Test Split (Stratified 80/20)
   |
Preprocessing (ColumnTransformer)
   |-- Numeric  : Median Imputation → StandardScaler
   └-- Categorical: Most Frequent Imputation → OrdinalEncoder
   |
Model Training
   |-- Logistic Regression  (Interpretable Baseline)
   |-- Random Forest        (Ensemble Baseline)
   └-- XGBoost              (Selected Best Model)
   |
Hyperparameter Tuning (RandomizedSearchCV + Stratified 3-Fold CV)
   |
Model Evaluation (Holdout Set + 5-Fold Cross-Validation)
   |
SHAP Explainability (TreeExplainer)
   |
Serialised Pipeline  →  saved locally (not tracked in repo)
```

---

## 📂 Dataset Description

The dataset contains **~37,000 anonymised customer records** with behavioural, transactional, and membership attributes collected from a retail/subscription platform.

| Feature | Type | Description |
|---|---|---|
| `points_in_wallet` | Numeric | Accumulated loyalty reward points |
| `avg_transaction_value` | Numeric | Customer's average spend per transaction |
| `avg_frequency_login_days` | Numeric | Average days between platform logins |
| `avg_time_spent` | Numeric | Average session duration per visit |
| `no_of_days_visited` | Numeric | Total platform visits in observation window |
| `membership_category` | Categorical | No Membership / Basic / Silver / Gold / Platinum / Premium |
| `region_category` | Categorical | City / Town / Village |
| `channel_code` | Categorical | Acquisition channel (Web, App, etc.) |
| `feedback` | Categorical | Customer's last recorded feedback sentiment |
| `past_complaint` | Categorical | Whether a complaint was previously raised |
| `complaint_status` | Categorical | Resolution status of the last complaint |
| `used_special_discount` | Categorical | Whether the customer used a discount |
| `offer_application_preference` | Categorical | Preference for receiving offers |
| `churn_risk_score` | Binary | **Target**: 1 = Churned, 0 = Retained |

**Target Distribution:** Near-balanced (~50/50), eliminating the need for SMOTE or aggressive resampling.

---

## 📦 Dataset Source

This dataset is sourced from a simulated retail customer behaviour dataset used for churn prediction research. It is publicly available on Kaggle and is included in this repository as `churn.csv` for convenience.

To access the original source directly, visit:
[https://www.kaggle.com/code/undersc0re/customer-churn/input](https://www.kaggle.com/code/undersc0re/customer-churn/input)

---

## 🧹 Data Cleaning and Preprocessing

Raw data quality issues were handled systematically before any modelling step:

**Issues identified:**
- `'?'` placeholder strings masking missing values across multiple columns
- `avg_frequency_login_days` stored as `object` dtype instead of numeric
- Identifier and datetime columns (`security_no`, `referral_id`, `joining_date`, `last_visit_time`) with no predictive signal

**Steps taken:**
1. Replaced all `'?'` values with `np.nan` for proper imputation
2. Coerced `avg_frequency_login_days` to numeric using `pd.to_numeric(errors='coerce')`
3. Dropped non-predictive identifier and timestamp columns
4. Separated features and target **before any preprocessing** to prevent data leakage

All subsequent preprocessing (imputation, scaling, encoding) was applied **exclusively within a fitted sklearn Pipeline**, guaranteeing that no information from the test set was used during training.

---

## 📊 Exploratory Data Analysis (EDA)

Key patterns surfaced during EDA that directly informed the modelling strategy:

**Churn by Membership Tier:** Customers with no membership or basic membership exhibited the highest churn rates (approximately 90 to 95%), while premium-tier customers were nearly fully retained. This confirmed `membership_category` as a high-signal feature.

**Wallet Points vs. Churn:** Churned customers held significantly fewer loyalty points on average than retained customers, suggesting the rewards programme is a meaningful retention mechanism.

**Transaction Value & Engagement:** Lower average spend and reduced login frequency were both strongly correlated with churn, indicating disengagement precedes cancellation by a measurable window.

**Feedback Sentiment:** Customers who rated their experience as "Poor Customer Support" or "Too Many Pop-ups" churned at significantly higher rates than those who gave positive feedback.

---

## ⚙️ Feature Engineering

No manual feature construction was required. Instead, engineering was embedded directly in the preprocessing pipeline:

- **Numeric features:** Median imputation (robust to outliers) → `StandardScaler`
- **Categorical features:** Most-frequent imputation → `OrdinalEncoder` with `handle_unknown='use_encoded_value'` to handle unseen categories at inference time
- **Pipeline architecture:** All transformations were bundled in a `ColumnTransformer` and wrapped within `sklearn Pipeline` objects to prevent leakage and ensure single-call production inference

---

## 🤖 Machine Learning Models Used

Three models were trained and compared, each wrapped in a full preprocessing + classifier `Pipeline`:

### 1. Logistic Regression *(Interpretable Baseline)*
- `class_weight='balanced'` to account for any residual class imbalance
- `max_iter=1000` to ensure convergence
- Serves as a linearity benchmark

### 2. Random Forest *(Ensemble Baseline)*
- 100 estimators, `class_weight='balanced'`
- Captures non-linear interactions without feature scaling dependency
- Used for feature importance triangulation against XGBoost

### 3. XGBoost *(Selected Best Model)*
- Gradient-boosted trees with `logloss` evaluation metric
- Hyperparameters tuned via `RandomizedSearchCV` (20 iterations, 3-Fold Stratified CV)
- Optimised directly on F1 score, the metric that balances false negatives (missed churners) against false positives (wasted retention spend)
- Tuned model (F1: **94.02%**) marginally outperformed the baseline (F1: 94.01%) and was selected as the final deployed pipeline

---

## 📈 Model Evaluation Metrics

### Holdout Test Set Performance (80/20 Stratified Split)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 77.56% | 79.06% | 79.62% | 79.34% | 83.29% |
| Random Forest | 93.16% | 92.91% | 94.58% | 93.74% | 97.55% |
| **XGBoost** | **93.47%** | **93.29%** | **94.75%** | **94.01%** | **97.60%** |

### Stratified 5-Fold Cross-Validation (Mean ± Std)

Cross-validation confirmed that XGBoost's performance is **stable and generalisable**, not an artefact of a favourable random seed.

| Model | CV F1 (Mean ± Std) | CV ROC-AUC (Mean ± Std) |
|---|---|---|
| Logistic Regression | 78.70% ± 0.38% | 82.47% ± 0.36% |
| Random Forest | 93.73% ± 0.43% | 97.44% ± 0.11% |
| **XGBoost** | **93.80% ± 0.31%** | **97.55% ± 0.13%** |

**Why F1 over Accuracy?**
In churn prediction, a false negative (missing a true churner) is costly; the company loses a customer entirely. A false positive (flagging a retained customer) wastes a targeted retention offer. F1 penalises both equally, making it the correct optimisation target for this business problem.

---

## 💡 Key Insights and Business Impact

### Top Churn Drivers (SHAP-Verified)

SHAP (SHapley Additive exPlanations) was used to provide directional, per-prediction feature attribution, going beyond aggregate importance scores.

| Rank | Feature | Direction | Business Meaning |
|---|---|---|---|
| 1 | `points_in_wallet` | Low → High churn | Loyalty rewards are the single strongest retention lever |
| 2 | `membership_category` | Basic/None → High churn | Premium membership dramatically reduces churn risk |
| 3 | `avg_transaction_value` | Low → High churn | High-value customers have greater product stickiness |
| 4 | `feedback` | Negative → High churn | Poor support experience is a leading churn indicator |
| 5 | `avg_frequency_login_days` | Less active → High churn | Platform disengagement precedes cancellation |

### Business Recommendations

**1. 💰 Launch a wallet top-up campaign.** Target customers with `points_in_wallet` below the model-identified risk threshold with bonus points or cashback to increase perceived loyalty programme value.

**2. 🎖️ Offer tiered membership upgrades.** Customers in "No Membership" and "Basic Membership" categories churn at approximately 90 to 95%. Subsidised trial upgrades to Silver or Gold could significantly shift retention rates.

**3. 🛠️ Prioritise complaint resolution.** Customers with unresolved complaints and negative feedback are at immediate churn risk. A dedicated fast-track resolution queue for flagged customers would have high ROI.

**4. ⚡ Automate real-time churn scoring.** The serialised pipeline can score a new customer record in milliseconds. Integrating this into the CRM or data warehouse enables daily automated risk scoring and proactive outreach.

**Estimated Business Value:** If the model flags 1,000 at-risk customers monthly and a targeted retention campaign recovers 30% of them at an average LTV of ₦500,000 per customer, the recovered revenue is ₦150,000,000/month, from a model trained on existing data.

---

## 🛠️ Technologies and Tools Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Manipulation** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn`, `xgboost` |
| **Explainability** | `shap` |
| **Visualisation** | `matplotlib`, `seaborn` |
| **Model Serialisation** | `joblib` |
| **Hyperparameter Tuning** | `RandomizedSearchCV` |
| **Validation** | `StratifiedKFold` |
| **Development Environment** | Jupyter Notebook / Google Colab |
| **Version Control** | Git / GitHub |

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── churn_prediction.ipynb        # Main notebook: full end-to-end pipeline
├── churn.csv                     # Raw dataset (37k customer records)
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
├── visuals/                      # Exported chart images
│   ├── target_distribution.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance_rf.png
│   ├── feature_importance_xgb.png
│   ├── shap_summary_beeswarm.png
│   ├── shap_summary_bar.png
│   └── shap_dependence_wallet.png
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/profpius/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
xgboost>=2.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
jupyter>=1.0
```

### 3. Launch the Notebook

```bash
jupyter notebook churn_prediction.ipynb
```

Or open directly in Google Colab using the badge at the top of this README.

### 4. Run Inference on a New Customer

> **Note:** Run the notebook first to generate `churn_model_pipeline.pkl` locally, then use the snippet below.

```python
import joblib
import pandas as pd

pipeline = joblib.load('churn_model_pipeline.pkl')

new_customer = pd.DataFrame([{
    'region_category'            : 'City',
    'membership_category'        : 'Basic Membership',
    'joining_month'              : 3,
    'channel_code'               : 'Web',
    'avg_frequency_login_days'   : 30,
    'points_in_wallet'           : 150.0,
    'used_special_discount'      : 'No',
    'offer_application_preference': 'No',
    'past_complaint'             : 'Yes',
    'complaint_status'           : 'Unsolved',
    'feedback'                   : 'Poor Customer Support',
    'avg_transaction_value'      : 20000,
    'avg_time_spent'             : 8.0,
    'no_of_days_visited'         : 5,
}])

pred  = pipeline.predict(new_customer)[0]
prob  = pipeline.predict_proba(new_customer)[0][1]

print(f"Churn Prediction : {'CHURN' if pred == 1 else 'NO CHURN'}")
print(f"Churn Probability: {prob * 100:.1f}%")
```

---

## 🔭 Future Improvements

- **Threshold Optimisation:**  Apply a custom decision threshold (e.g., 0.35 instead of 0.50) calibrated to the business cost ratio of false negatives vs. false positives
- **Feature Engineering:**  Engineer time-based features from `joining_date` and `last_visit_time` (e.g., recency, days since last visit) which were dropped in this iteration
- **Calibrated Probabilities:**  Apply Platt Scaling or Isotonic Regression to ensure predicted probabilities are well-calibrated for reliable risk scoring
- **REST API Deployment:**  Wrap the serialised pipeline in a FastAPI service for real-time scoring via HTTP requests
- **Dashboard Integration:**  Build a Streamlit or Power BI dashboard surfacing high-risk customer lists for the retention team
- **Drift Monitoring:**  Implement data drift detection using Evidently AI to alert when the incoming customer distribution shifts from the training distribution

---

## 👤 Author

**Victor Pius**
Data Scientist & Data Analyst

I build end-to-end machine learning solutions with a focus on **production readiness**, **business interpretability**, and **robust evaluation**. This project demonstrates proficiency across the full data science lifecycle: from raw data ingestion and leak-proof preprocessing, through model selection and rigorous cross-validation, to SHAP-based explainability and deployment-ready serialisation.

📧 Connect on [LinkedIn](https://www.linkedin.com/in/victor-pius-4061a9332) · 💻 View all projects on [GitHub](https://github.com/profpius)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*If this project was useful, please consider giving it a ⭐ as it helps others discover it.*
