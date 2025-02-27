import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd


def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    data_dict = pd.read_csv('data_dictionary.csv')
    return train, test, data_dict

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


def stratified_c_index(y_true, y_pred, groups):
    """
    Calculate stratified concordance index across different groups
    
    Args:
        y_true: DataFrame with 'efs_time' and 'efs' columns
        y_pred: Predicted risk scores
        groups: Group labels (races)
    
    Returns:
        float: stratified c-index (mean - std of group-wise c-indices)
    """
    group_cindices = []
    
    for group in np.unique(groups):
        mask = (groups == group)
        if mask.sum() > 0:
            c_index = concordance_index(
                event_times=y_true.loc[mask, 'efs_time'],
                predicted_scores=y_pred[mask],
                event_observed=y_true.loc[mask, 'efs']
            )
            group_cindices.append(c_index)
    
    return np.mean(group_cindices) - np.std(group_cindices)

class StratifiedCIndexObjective:
    def __init__(self, y_true, groups):
        self.y_true = y_true
        self.groups = groups

    def __call__(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple:
        """
        Custom objective function for XGBoost
        """
        c_index = stratified_c_index(self.y_true, predt, self.groups)
        # Convert to minimization problem (XGBoost minimizes the objective)
        grad = np.zeros_like(predt)  # Approximate gradient
        hess = np.ones_like(predt)   # Approximate hessian
        return grad, hess

def train_race_specific_models(train_df, test_df):
    """
    Train separate XGBoost models for each race group
    """
    # Identify race dummy columns
    race_cols = [col for col in train_df.columns if col.startswith('race_group_')]
    feature_cols = [col for col in train_df.columns 
                   if col not in ['ID', 'efs', 'efs_time'] + race_cols]
    
    # Clean feature names for XGBoost
    feature_name_map = {col: col.replace('[', '_').replace(']', '_').replace('<', '_') 
                       for col in feature_cols}
    
    models = {}
    predictions = {}
    
    for race_col in race_cols:
        race = race_col.replace('race_group_', '')
        print(f"\nTraining model for race: {race}")
        
        # Filter data for specific race
        race_mask = (train_df[race_col] == 1)
        X_train_race = train_df.loc[race_mask, feature_cols].copy()
        y_train_race = train_df.loc[race_mask, ['efs_time', 'efs']]
        
        # Rename columns to be XGBoost compatible
        X_train_race.rename(columns=feature_name_map, inplace=True)
        
        # Create DMatrix with cleaned feature names
        dtrain = xgb.DMatrix(X_train_race, label=y_train_race['efs_time'])
                
        # Parameters
        params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'eta': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3
        }
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            verbose_eval=100
        )
        
        models[race] = model
        
        # Make predictions for this race group
        race_mask_test = (test_df[f'race_group_{race}'] == race)
        X_test_race = test_df.loc[race_mask_test, feature_cols]
        X_test_race.rename(columns=feature_name_map, inplace=True)
        dtest = xgb.DMatrix(X_test_race)
        predictions[race] = model.predict(dtest)
    
    return models, predictions

def create_ensemble_submission(predictions, test_df):
    """
    Combine predictions from race-specific models
    """
    final_predictions = pd.DataFrame()
    final_predictions['ID'] = test_df['ID']
    
    # Combine predictions
    all_preds = []
    for race, preds in predictions.items():
        race_mask = (test_df[f'race_group_{race}'] == 1)
        temp_preds = np.zeros(len(test_df))
        temp_preds[race_mask] = preds
        all_preds.append(temp_preds)
    
    final_predictions['prediction'] = np.mean(all_preds, axis=0)
    
    return final_predictions

# Main execution
def main():
    # Load and preprocess data
    train_df, test_df, _ = load_data()
    train_processed, test_processed = preprocess_data(train_df, test_df)
    
    # Train race-specific models
    models, predictions = train_race_specific_models(train_processed, test_processed)
    
    # Create submission
    submission = create_ensemble_submission(predictions, test_processed)
    submission.to_csv('submission.csv', index=False)


print("Starting main execution")
main()

print("Execution completed")