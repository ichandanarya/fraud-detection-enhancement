# Importing required libraries for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np


# Function to split dataset into training and testing sets
def split_data(df):
    # Separating features (X) and target variable (y)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Splitting data (80% train, 20% test)
    # stratify ensures same fraud ratio in both sets
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Function to train baseline model (Logistic Regression)
def train_baseline_model(df):
    # Splitting data
    X_train, X_test, y_train, y_test = split_data(df)

    # Initializing Logistic Regression model
    # - max_iter: ensures convergence
    # - solver: suitable for small/medium datasets
    # - class_weight='balanced': handles class imbalance
    model = LogisticRegression(
        max_iter=2000,
        solver='liblinear',
        class_weight='balanced'
    )

    # Training the model
    model.fit(X_train, y_train)

    # Getting predicted probabilities for fraud class
    y_prob = model.predict_proba(X_test)[:,1]

    # Returning model and evaluation data
    return model, X_test, y_test, y_prob


# Function to train enhanced model (XGBoost)
def train_xgb_model(df):
    # Splitting data
    X_train, X_test, y_train, y_test = split_data(df)

    # Calculating imbalance ratio (non-fraud / fraud)
    # This helps XGBoost focus more on minority class (fraud)
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Initializing XGBoost model with tuned parameters
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )

    # Training the model
    model.fit(X_train, y_train)

    # Getting predicted probabilities for fraud class
    y_prob = model.predict_proba(X_test)[:, 1]

    # Returning model and evaluation data
    return model, X_test, y_test, y_prob


# Function to evaluate model performance
def evaluate_model(y_test, y_prob):
    # Printing ROC-AUC score to measure model performance
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# Function to calculate KS statistic
def ks_statistic(y_true, y_prob):
    # Creating DataFrame with actual labels and predicted probabilities
    df = pd.DataFrame({'y': y_true, 'prob': y_prob})
    
    # Sorting by predicted probability (highest risk first)
    df = df.sort_values(by='prob', ascending=False)

    # Calculating cumulative distribution for fraud (event)
    df['cum_event'] = np.cumsum(df['y']) / sum(df['y'])

    # Calculating cumulative distribution for non-fraud (non-event)
    df['cum_non_event'] = np.cumsum(1 - df['y']) / sum(1 - df['y'])

    # Returning maximum difference between the two distributions
    return max(abs(df['cum_event'] - df['cum_non_event']))