# 📊 Customer Churn Prediction Using Machine Learning

A machine learning project to predict customer churn using a dataset of 36,992 customer records. The goal was to identify customers who are likely to leave and uncover the key factors driving churn, so the business can take action before losing them.

---

## 🎯 Problem Statement

Customer churn is one of the most costly problems a business can face. Retaining an existing customer is far cheaper than acquiring a new one. This project builds a machine learning model that predicts which customers are at risk of leaving, and provides actionable recommendations the business can use to improve retention.

---

## 🗂️ Dataset

- **Records:** 36,992 customers
- **Features:** 23 columns including demographics, membership details, transaction behaviour, and feedback
- **Target:** `churn_risk_score` — binary (1 = Churned, 0 = Stayed)
- **Churn Rate:** 54.1%

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Dropped non-predictive columns — `security_no` and `referral_id` are unique identifiers with no predictive value; `joining_date` and `last_visit_time` are raw datetime strings that require feature engineering before they are useful
- Replaced `?` placeholders with NaN
- Imputed missing values — median for numeric columns, most frequent value for categorical columns
- Label encoded all categorical columns

### 2. Model Training
Three models were trained and compared:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | ROC-AUC (%) |
|---|---|---|---|---|---|
| Logistic Regression | 77.59 | 79.09 | 79.64 | 79.36 | 83.29 |
| Random Forest | 93.23 | 92.77 | 94.88 | 93.81 | 97.60 |
| **XGBoost** | **93.45** | **93.37** | **94.60** | **93.98** | **97.63** |

### 3. Model Selection
**XGBoost** was selected as the best model. It achieved the highest F1 Score (93.98%), Accuracy (93.45%), and Precision (93.37%). For churn prediction, F1 matters most because missing a real churner is more costly to the business than a false alarm. XGBoost consistently outperformed the other models across the metrics that matter most for this problem.

### 4. Hyperparameter Tuning
RandomizedSearchCV was used across 20 iterations to find the best combination of parameters for XGBoost. The best parameters found were:

| Parameter | Value |
|---|---|
| n_estimators | 300 |
| max_depth | 5 |
| learning_rate | 0.01 |
| subsample | 0.7 |
| colsample_bytree | 0.7 |

**Tuning Result:**
The tuned model achieved an F1 of 93.94%, marginally below the default model's 93.98%, suggesting the initial parameters were already well-suited to this dataset. The original XGBoost model was therefore retained as the final model.

---

## 🔑 Key Findings

### Top Churn Predictors
1. **points_in_wallet** — The strongest predictor by far. Customers who churned had an average of 630 wallet points compared to 740 for those who stayed. Low engagement with the loyalty programme is a strong early warning signal.
2. **membership_category** — Churn rate drops dramatically as membership tier increases. No Membership (~90%) and Basic Membership (~95%) customers churn the most, while Premium membership customers are almost completely loyal.
3. **avg_transaction_value** — Customers with lower average spend are more likely to leave.
4. **feedback** — Negative customer feedback strongly correlates with churn.
5. **avg_frequency_login_days** — Less frequent users are at significantly higher churn risk.

---

## 💡 Business Recommendations

**1. Re-engage Customers with Low Wallet Points**

I found that customers with lower wallet points were more likely to churn. This tells me these customers are not actively engaging with the loyalty programme. The business can address this by sending personalised offers or bonus point campaigns to customers whose points drop below a certain level.

**2. Focus on No Membership and Basic Membership Customers**

My analysis showed that customers with no membership or basic membership had the highest churn rates was around 90% and 95% respectively. Customers on premium plans barely churned at all. I would recommend the business introduce upgrade incentives to move low-tier customers into higher membership categories, as this would give them more reasons to stay.

**3. Act on Negative Feedback Early**

Customer feedback was one of the top predictors of churn in my model. Customers who expressed dissatisfaction were significantly more likely to leave. A simple fix would be for the customer service team to follow up with unhappy customers immediately before they make the decision to leave.

---

## 🛠️ Tools & Libraries

- **Python**
- **Pandas & NumPy** — data manipulation
- **Scikit-learn** — preprocessing, Logistic Regression, Random Forest, model evaluation
- **XGBoost** — final model
- **Matplotlib & Seaborn** — visualisations
- **Jupyter Notebook**

---

## 🚀 Future Improvements


- Extract useful features from `joining_date` (customer tenure) and `last_visit_time` (hour of activity) instead of dropping them
- Tune the decision threshold beyond the default 0.5 depending on business tolerance for false alarms
- Apply cross validation to ensure results are consistent across different data splits
- Run RandomizedSearchCV with more iterations to explore a wider range of parameter combinations

---

## 🧑‍💻 Author

**Victor Pius**

Data Scientist

[

![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)(https://www.linkedin.com/in/victor-pius-4061a9332)

![Github](https://img.shields.io/badge/GitHub-Follow-green)
(https://github.com/profpius)
