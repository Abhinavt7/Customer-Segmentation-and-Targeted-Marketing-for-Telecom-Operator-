# Preprocessing module for data cleaning and feature engineering

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

#def clean_data(df):
    # Remove duplicate rows
    #df = df.drop_duplicates()
    # Reset index after removing duplicates
    #df = df.reset_index(drop=True)
    #return df

def handle_missing_values(df):
    # Handling missing values by filling with median for numerical and mode for categorical
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def remove_outliers(df, columns, z_threshold=3):
    # Removing outliers using Z-score method
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[columns]))
    df_clean = df[(z_scores < z_threshold).all(axis=1)]
    return df_clean

def encode_categorical(df):
    # Encoding categorical variables using Label Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

def scale_features(df):
    #scaling features using StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Saving the scaler for future use
    import joblib
    joblib.dump(scaler, '../models/scaler.pkl')
    
    return scaled_data
