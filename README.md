# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, with a large number of normal transactions and very few fraudulent transactions. The goal is to build a model that accurately identifies fraud while minimizing false positives.

---

## Dataset

- **Source:** Kaggle Credit Card Fraud Detection Dataset  
- **Size:** ~284,000 transactions  
- **Features:**  
  - `Time` : Seconds elapsed between this transaction and the first transaction  
  - `V1`–`V28` : PCA-transformed features  
  - `Amount` : Transaction amount  
  - `Class` : Target variable (0 = Normal, 1 = Fraud)  

**Class Distribution:**

- Normal transactions: ~99%  
- Fraud transactions: ~0.2%  

---

## Exploratory Data Analysis (EDA)

- Checked **distribution** of features using histograms and KDE plots.  
- Visualized **fraud vs normal** over `Time` and `Amount`.  
- Identified outliers in `Amount` and PCA features (`V1`–`V28`).  
- Confirmed **no missing values** and dropped duplicates.  
- Created new features:  
  - `Hour` from `Time`  
  - `Amount_log` (log transformation of Amount)  
  - Scaled `Amount` and `Hour` using StandardScaler  

---

## Handling Imbalanced Data

- Applied **SMOTE** (Synthetic Minority Oversampling Technique) on training data only to balance the classes.  
- After SMOTE: Fraud class increased to match the number of normal transactions in the training set.  

---

## Models Built

1. **Logistic Regression (Baseline)**  
   - Initial model on imbalanced data  
   - F1-score for fraud class: ~0.34  
   - Poor performance due to extreme imbalance  

2. **Decision Tree**  
   - Applied SMOTE  
   - Parameters: `max_depth=10`, `min_samples_leaf=10`, `class_weight='balanced'`  
   - Better performance than logistic regression  

3. **Random Forest** *(Best Model)*  
   - Parameters: `n_estimators=50`, `max_depth=12`, `class_weight='balanced'`, `n_jobs=-1`  
   - SMOTE applied on training data  
   - **Metrics:**
     - Precision: 0.73  
     - Recall: 0.81  
     - F1-score: 0.77  
     - ROC-AUC: 0.974  
   - Reasonably detects most frauds with low false positives  

 

---

## Key Observations

- **Imbalanced data** makes metrics like accuracy misleading.  
- **F1-score** is the key metric for fraud detection.  
- Random Forest with SMOTE is **effective** for imbalanced datasets.  
- Model successfully balances detection of frauds (high recall) and false alarms (precision).  

---

## How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost


DataSet link
https://www.kaggle.com/mlg-ulb/creditcardfraud