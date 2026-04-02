# Importing numpy for numerical operations (used for log transformation)
import numpy as np

# Function to create new features from existing data
def create_features(df):
    
    # Creating 'hour' feature from 'Time'
    # Converts time (in seconds) into hours to capture transaction timing patterns
    df['hour'] = df['Time'] // 3600
    
    # Applying log transformation to 'Amount'
    # Helps reduce skewness caused by very large transaction values
    df['amount_log'] = np.log1p(df['Amount'])
    
    # Returning the updated dataframe with new features
    return df