# Importing StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler


# Function to preprocess the dataset
def preprocess_data(df):
    
    # Initializing scaler to normalize feature values
    scaler = StandardScaler()

    # Separating features (X) and target variable (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Scaling all feature columns
    # This ensures all features are on the same scale and improves model performance
    X_scaled = scaler.fit_transform(X)

    # Converting scaled data back to DataFrame
    # This helps retain column names for better readability
    import pandas as pd
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Adding target column back to the scaled dataset
    X_scaled['Class'] = y.values

    # Returning the preprocessed dataset
    return X_scaled