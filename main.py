# Required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    data_dict = pd.read_csv('data_dictionary.csv')
    return train, test, data_dict


# 2. Initial EDA
def explore_data(train_df, plot_survival_curves=True):
    print("Dataset Shape:", train_df.shape)
    
    # Ensure efs_time and efs are numeric
    train_df['efs_time'] = pd.to_numeric(train_df['efs_time'], errors='coerce')
    train_df['efs'] = pd.to_numeric(train_df['efs'], errors='coerce')
    
    # Separate numerical and categorical columns
    numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns
    
    print("\nNumerical Features:", len(numerical_cols))
    print("Categorical Features:", len(categorical_cols))
    
    # Numerical Features Summary
    print("\nNumerical Features Summary:")
    print(train_df[numerical_cols].describe())
    
    # Categorical Features Summary
    print("\nCategorical Features Value Counts:")
    for col in categorical_cols:
        if col != 'ID':  # Skip ID column
            print(f"\n{col}:")
            print(train_df[col].value_counts().head())
    
    # Basic survival analysis
    kmf = KaplanMeierFitter()
    
    # First, plot overall survival
    plt.figure(figsize=(10, 6))
    kmf.fit(
        durations=train_df['efs_time'],
        event_observed=train_df['efs'],
        label='Overall'
    )
    kmf.plot()
    plt.title('Overall Survival Curve')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.show()
    
    # Plot survival curves for important categorical variables

    if plot_survival_curves:
        for cat_var in categorical_cols:
            if cat_var not in ['ID'] and len(train_df[cat_var].unique()) <= 5:  # Only plot if 5 or fewer categories
                plt.figure(figsize=(10, 6))
                for val in train_df[cat_var].unique():
                    mask = (train_df[cat_var] == val)
                    if mask.sum() > 0:  # Only fit if there are samples in this category
                        kmf.fit(
                            durations=train_df[mask]['efs_time'],
                            event_observed=train_df[mask]['efs'],
                            label=f'{cat_var}={val}'
                        )
                        kmf.plot()
                plt.title(f'Survival Curves by {cat_var}')
                plt.xlabel('Time')
                plt.ylabel('Survival Probability')
                plt.show()
    
    # Correlation heatmap for numerical features
    numerical_cols = [col for col in numerical_cols if col not in ['ID']]
    if len(numerical_cols) > 0:
        plt.figure(figsize=(12, 8))
        sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Numerical Features Correlations')
        plt.tight_layout()
        plt.show()

# 3. Data Preprocessing
def preprocess_data(train_df, test_df):
    # Make copies to avoid modifying original data
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Ensure efs_time and efs are numeric
    train_df['efs_time'] = pd.to_numeric(train_df['efs_time'], errors='coerce')
    train_df['efs'] = pd.to_numeric(train_df['efs'], errors='coerce')
    
    # Combine for consistent preprocessing
    all_data = pd.concat([train_df, test_df], axis=0)
    
    # Identify column types
    numerical_cols = all_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = all_data.select_dtypes(include=['object', 'category']).columns
    
    # Handle missing values for numerical columns
    for col in numerical_cols:
        if col not in ['ID', 'efs', 'efs_time']:
            all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
            all_data[col] = all_data[col].fillna(all_data[col].median())
    
    # Handle missing values for categorical columns
    for col in categorical_cols:
        if col != 'ID':
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    
    # Handle categorical variables
    categorical_cols = [col for col in categorical_cols if col != 'ID']
    all_data = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)
    
    # Scale numerical features
    numerical_cols = [col for col in numerical_cols 
                     if col not in ['ID', 'efs', 'efs_time']]
    
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        all_data[numerical_cols] = scaler.fit_transform(all_data[numerical_cols])
    
    # Split back to train and test
    train_processed = all_data[all_data['ID'].isin(train_df['ID'])]
    test_processed = all_data[all_data['ID'].isin(test_df['ID'])]
    
    return train_processed, test_processed

# ... existing code ...

# 4. Model Building using Cox Proportional Hazards
def build_cph_model(train_processed):
    # Initialize Cox model
    cph = CoxPHFitter()
    
    # Prepare data for Cox model
    columns_for_model = [col for col in train_processed.columns 
                        if col not in ['ID', 'efs', 'efs_time']]
    
    # Fit model
    cph.fit(
        train_processed,
        duration_col='efs_time',
        event_col='efs',
        #covariates=columns_for_model,
        show_progress=True
    )
    
    return cph

# 5. Make Predictions
def make_predictions(model, test_processed):
    # Generate risk scores
    risk_scores = model.predict_partial_hazard(test_processed)
    
    # Prepare submission
    submission = pd.DataFrame({
        'ID': test_processed['ID'],
        'prediction': risk_scores
    })
    
    return submission
