# 🚀 Fraud Detection Model Enhancement for Financial Applications

---

## 📌 Project Overview

This project focuses on building an improved fraud detection system for financial transactions. The goal is to enhance a basic (legacy) model by addressing real-world challenges such as class imbalance, changing data patterns, and model performance degradation over time.

Instead of just building a model, this project simulates how fraud detection systems work in real banking environments — including evaluation, monitoring, and continuous improvement.

---

## 🎯 Objective

* Detect fraudulent transactions with high accuracy
* Handle highly imbalanced data
* Improve fraud detection compared to a baseline model
* Simulate real-world challenges like distribution shift
* Build a monitoring system to track model performance

---

## 📊 Dataset

* Credit Card Fraud Detection Dataset
* Contains anonymized transaction features (V1–V28, Time, Amount)

**Target variable:**

* `0` → Non-Fraud
* `1` → Fraud

⚠️ Note: The dataset is highly imbalanced, which reflects real-world fraud scenarios.

---

## 🧠 Project Workflow

1. **Exploratory Data Analysis (EDA)**

   * Fraud vs Non-Fraud distribution
   * Transaction patterns
   * Feature relationships

2. **Data Preprocessing**

   * Feature scaling
   * Train-test split
   * Handling imbalance using SMOTE

3. **Feature Engineering**

   * Time-based features (hour)
   * Log transformation of transaction amount

4. **Baseline Model**

   * Logistic Regression
   * Shows limitations in fraud detection

5. **Enhanced Model**

   * XGBoost (Gradient Boosting)
   * Improved performance and fraud detection

6. **Model Evaluation**

   * ROC-AUC
   * KS Statistic
   * Fraud Capture Rate (Top %)

7. **Distribution Shift Simulation**

   * Artificially modified data to simulate real-world drift
   * Observed performance degradation

8. **Model Monitoring**

   * PSI (Population Stability Index)
   * Performance tracking over time

9. **Explainability**

   * SHAP values for model interpretation

---

## 📈 Results

| Metric             | Baseline Model | Enhanced Model         |
| ------------------ | -------------- | ---------------------- |
| Accuracy           | High           | Slightly Lower         |
| Recall (Fraud)     | Low ❌          | High ✅                 |
| ROC-AUC            | Moderate       | High 🚀                |
| Fraud Capture Rate | Low            | Significantly Improved |

👉 The enhanced model performs much better in identifying fraud cases while maintaining overall stability.

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn
* SHAP
* Streamlit

---

## ⚠️ Challenges

* Highly imbalanced dataset
* Limited real-world features (no customer data)
* Simulating distribution shift
* Balancing performance and interpretability

---

## 🔍 Key Learnings

* Accuracy alone is not sufficient for fraud detection
* Handling class imbalance is critical
* Models degrade over time due to data drift
* Monitoring systems are essential in production

---

## 🚀 Future Improvements

* Add customer-level and external data
* Build real-time fraud detection system
* Deploy model using FastAPI

---

## 📂 Project Structure

```
fraud-detection-enhancement/
│
├── data/
├── notebooks/
├── src/
├── models/
├── dashboard/
├── api/
├── reports/
└── README.md
```

---

## ⚙️ Setup & Run the Project

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/ichandanarya/fraud-detection-enhancement.git
cd fraud-detection-enhancement
```

---

### 🔹 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate environment:

**Windows (PowerShell):**

```bash
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

---

### 🔹 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 4. Dataset Setup

Download dataset from:

👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading:

* Place `creditcard.csv` inside the `data/` folder

---

### 🔹 5. Run Jupyter Notebooks (Optional)

```bash
jupyter notebook
```

Go to:

```
notebooks/
```

---

### 🔹 6. Run Streamlit Dashboard(Visual Studio Code)

```bash
cd dashboard
streamlit run streamlit_app.py
```

👉 This will open the dashboard in your browser.

## 👨‍💻 Author

**Chandan Arya**

---

## ⭐ Final Note

This project is not just about building a machine learning model — it is about understanding how fraud detection systems are designed, improved, and maintained in real-world financial environments.

If you found this project useful, feel free to ⭐ the repository!
