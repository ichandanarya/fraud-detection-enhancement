# Importing numpy for numerical operations
import numpy as np


# Function to simulate distribution shift in the dataset
def simulate_shift(X):
    # Creating a copy to avoid modifying original data
    X_shifted = X.copy()
    
    # Checking if 'Amount' column exists before applying transformation
    # This prevents errors if the column is missing
    if 'Amount' in X_shifted.columns:
        # Increasing transaction amounts to simulate real-world data changes
        # This mimics scenarios where user behavior or market conditions change
        X_shifted['Amount'] *= 1.5
    
    # Returning the modified dataset
    return X_shifted


# Function to calculate PSI (Population Stability Index)
def calculate_psi(expected, actual, bins=10):
    
    # Creating equal-width bins between 0 and 1
    # Used to group probability values into segments
    breakpoints = np.linspace(0, 1, bins + 1)

    # Calculating distribution of original (expected) data
    # Converted into proportions instead of raw counts
    expected_bins = np.histogram(expected, bins=breakpoints)[0] / len(expected)

    # Calculating distribution of shifted (actual) data
    actual_bins = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Applying PSI formula
    # Adding small value (1e-6) to avoid division by zero errors
    psi = np.sum((actual_bins - expected_bins) *
                 np.log((actual_bins + 1e-6) / (expected_bins + 1e-6)))

    # Returning PSI value
    return psi