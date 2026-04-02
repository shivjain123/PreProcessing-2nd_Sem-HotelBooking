
#Imports
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer

# ========================= Pipelines ========================= #

### From Task 1 ###

# Select a minimal set of features for baseline
baseline_numeric = ['lead_time', 'adr', 'stays_in_weekend_nights',
                    'stays_in_week_nights', 'adults', 'total_of_special_requests']
baseline_categorical = ['hotel', 'meal', 'market_segment']
target = 'is_canceled'

numeric_transformer_base = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # median is robust to outliers
    ('scaler', StandardScaler())  # scale features to similar range (not required for RF but keeps pipeline generic, so it can be reused)
])

categorical_transformer_base = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # fill missing categories with mode
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    # one-hot encode categories; ignore unseen values in test to avoid crash since real-world data can introduce classes in test not seen in training.
])

# Apply respective preprocessing to numeric vs categorical columns, then combine
preprocessor_base = ColumnTransformer([
    ('num', numeric_transformer_base, baseline_numeric),
    ('cat', categorical_transformer_base, baseline_categorical)
])

# Full pipeline → preprocessing + model in one (prevents data leakage)
baseline_pipeline = Pipeline([
    ('preprocessor', preprocessor_base),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

### From Task 5 ###

# Select relevant features for full pipeline
pipeline_numeric  = ['lead_time', 'adr', 'stays_in_week_nights',
                     'stays_in_weekend_nights', 'total_of_special_requests',
                     'previous_cancellations', 'booking_changes']
pipeline_categorical = ['hotel', 'meal', 'market_segment', 'deposit_type']

log_transformer = FunctionTransformer(np.log1p, validate=True)

# Numeric pipeline: log transform → impute → robust scale
numeric_pipeline = Pipeline([
    ('log',     log_transformer),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  RobustScaler())
])

# Categorical pipeline: impute → one-hot encode
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine with ColumnTransformer
preprocessor5 = ColumnTransformer([
    ('num', numeric_pipeline,      pipeline_numeric),
    ('cat', categorical_pipeline,  pipeline_categorical)
])

# Full pipeline with model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor5),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)) #n_jobs = -1 lets training happen using all CPU cores, making it much faster.
])

# ========================= Helper Functions ========================= #

def safe_div(a, b, fill=0):
    """
    Safely perform division while avoiding division-by-zero errors.

    Parameters:
    a (array-like): Numerator
    b (array-like): Denominator
    fill (float): Value to replace where division by zero occurs

    Returns:
    np.array: Result of division
    """
    return np.where(b == 0, fill, a / b)


def add_total_guests(df):
    """
    Adds total_guests feature.
    """
    return df['adults'] + df['children'].fillna(0) + df['babies'].fillna(0)


def add_total_nights(df):
    """
    Adds total_nights feature.
    """
    return df['stays_in_weekend_nights'] + df['stays_in_week_nights']


def create_is_family(df):
    """
    Creates binary feature indicating if booking includes family.
    """
    return ((df['children'].fillna(0) + df['babies'].fillna(0)) > 0).astype(int)