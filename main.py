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
def explore_data(train_df):
    print("Dataset Shape:", train_df.shape)
    print("\nRacial Distribution:")
    print(train_df['recipient_race'].value_counts())
    
    # Survival Analysis by Race
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(12, 6))
    
    for race in train_df['recipient_race'].unique():
        mask = (train_df['recipient_race'] == race)
        kmf.fit(
            train_df[mask]['efs_time'],
            train_df[mask]['efs'],
            label=race
        )
        kmf.plot()
    
    plt.title('Kaplan-Meier Survival Curves by Race')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.show()

# 3. Data Preprocessing
def preprocess_data(train_df, test_df):
    # Combine for consistent preprocessing
    all_data = pd.concat([train_df, test_df], axis=0)
    
    # Handle categorical variables
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    all_data = pd.get_dummies(all_data, columns=categorical_cols)
    
    # Scale numerical features
    numerical_cols = all_data.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col not in ['ID', 'efs', 'efs_time']]
    
    scaler = StandardScaler()
    all_data[numerical_cols] = scaler.fit_transform(all_data[numerical_cols])
    
    # Split back to train and test
    train_processed = all_data[all_data['ID'].isin(train_df['ID'])]
    test_processed = all_data[all_data['ID'].isin(test_df['ID'])]
    
    return train_processed, test_processed

# 4. Model Building using Cox Proportional Hazards
def build_model(train_processed):
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
        covariates=columns_for_model,
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

# Main execution
def main():
    # Load data
    train_df, test_df, data_dict = load_data()
    
    # Explore data
    explore_data(train_df)
    
    # Preprocess data
    train_processed, test_processed = preprocess_data(train_df, test_df)
    
    # Build model
    model = build_model(train_processed)
    
    # Generate predictions
    submission = make_predictions(model, test_processed)
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    
if __name__ == "__main__":
    main()