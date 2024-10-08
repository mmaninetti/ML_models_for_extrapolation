from umap import UMAP
import pandas as pd
import numpy as np
import openml
from sklearn.linear_model import LogisticRegression 
import lightgbm as lgbm
import optuna
from sklearn.ensemble import RandomForestClassifier
from engression import engression
import torch
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
import random
import os
from pygam import LogisticGAM
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from utils import EarlyStopping, train, train_trans, train_no_early_stopping, train_trans_no_early_stopping
from torch.utils.data import TensorDataset, DataLoader
import re
import shutil
import gpboost as gpb
from sklearn.model_selection import train_test_split

#SUITE_ID = 336 # Regression on numerical features
SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

#task_id=361055
for task_id in benchmark_suite.tasks:  # iterate over all tasks in the benchmark suite

    if task_id==361276:
        continue

    if task_id == 361062 or task_id == 361068 or task_id == 361275:
        continue

    # Set the random seed for reproducibility
    N_TRIALS=100
    N_SAMPLES=100
    PATIENCE=40
    N_EPOCHS=1000
    
    BATCH_SIZE=1024
    seed=10
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    CHECKPOINT_PATH = f'CHECKPOINTS/RANDOM/task_{task_id}.pt'

    print(f"Task {task_id}")

    if (task_id==361055) or (task_id==361062) or (task_id==361063) or (task_id==361065) or (task_id==361068) or (task_id==361275):
        accuracy_gam = float("NaN")
    else:

        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()

        X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute)
        
        if len(X) > 15000:
            indices = np.random.choice(X.index, size=15000, replace=False)
            X = X.iloc[indices,]
            y = y[indices]

        # Remove categorical columns with more than 20 unique values and non-categorical columns with less than 10 unique values
        # Remove non-categorical columns with more than 70% of the data in one category from X_clean
        for col in [attribute for attribute, indicator in zip(attribute_names, categorical_indicator) if indicator]:
            if len(X[col].unique()) > 20:
                X = X.drop(col, axis=1)

        X_clean=X.copy()
        for col in [attribute for attribute, indicator in zip(attribute_names, categorical_indicator) if not indicator]:
            if len(X[col].unique()) < 10:
                X = X.drop(col, axis=1)
                X_clean = X_clean.drop(col, axis=1)
            elif X[col].value_counts(normalize=True).max() > 0.7:
                X_clean = X_clean.drop(col, axis=1)

        # Find features with absolute correlation > 0.9
        corr_matrix = X_clean.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

        # Drop one of the highly correlated features from X_clean
        X_clean = X_clean.drop(high_corr_features, axis=1)

        # Rename columns to avoid problems with LGBM
        X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        # Transform y to int type, to then be able to apply BCEWithLogitsLoss
        # Create a label encoder
        le = LabelEncoder()
        # Fit the label encoder and transform y to get binary labels
        y_encoded = le.fit_transform(y)
        # Convert the result back to a pandas Series
        y = pd.Series(y_encoded, index=y.index)


        # Apply random split 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)


        # Standardize the data
        mean_X_train_ = np.mean(X_train_, axis=0)
        std_X_train_ = np.std(X_train_, axis=0)
        X_train_ = (X_train_ - mean_X_train_) / std_X_train_
        X_val = (X_val - mean_X_train_) / std_X_train_

        mean_X_train = np.mean(X_train, axis=0)
        std_X_train = np.std(X_train, axis=0)
        X_train = (X_train - mean_X_train) / std_X_train
        X_test = (X_test - mean_X_train) / std_X_train


        # Convert data to PyTorch tensors
        X_train__tensor = torch.tensor(X_train_.values, dtype=torch.float32)
        y_train__tensor = torch.tensor(y_train_.values, dtype=torch.float32)
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        # Convert to use GPU if available
        if torch.cuda.is_available():
            X_train__tensor = X_train__tensor.cuda()
            y_train__tensor = y_train__tensor.cuda()
            X_train_tensor = X_train_tensor.cuda()
            y_train_tensor = y_train_tensor.cuda()
            X_val_tensor = X_val_tensor.cuda()
            y_val_tensor = y_val_tensor.cuda()
            X_test_tensor = X_test_tensor.cuda()
            y_test_tensor = y_test_tensor.cuda()

        # Create flattened versions of the data
        y_val_np = y_val.values.flatten()
        y_test_np = y_test.values.flatten()

        # Create TensorDatasets for training and validation sets
        train__dataset = TensorDataset(X_train__tensor, y_train__tensor)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create DataLoaders for training and validation sets
        train__loader = DataLoader(train__dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        d_out = 1  
        d_in=X_train_.shape[1]

        #### GAM model
        def gam_model(trial):

            # Define the search space for n_splines, lam, and spline_order
            n_splines=trial.suggest_int('n_splines', 10, 100)
            lam=trial.suggest_float('lam', 1e-3, 1e3, log=True)
            spline_order=trial.suggest_int('spline_order', 1, 5)
            
            # Create and train the model
            gam = LogisticGAM(n_splines=n_splines, spline_order=spline_order, lam=lam).fit(X_train_, y_train_)

            # Predict on the validation set and calculate the accuracy
            y_val_hat_gam = gam.predict(X_val)
            accuracy_gam = accuracy_score(y_val, y_val_hat_gam)

            return accuracy_gam

        # Create the sampler and study
        sampler_gam = optuna.samplers.TPESampler(seed=seed)
        study_gam = optuna.create_study(sampler=sampler_gam, direction='maximize')

        # Optimize the model
        study_gam.optimize(gam_model, n_trials=N_TRIALS)

        # Get the best parameters
        best_params = study_gam.best_params
        n_splines=best_params['n_splines']
        lam=best_params['lam']
        spline_order=best_params['spline_order']

        final_gam_model = LogisticGAM(n_splines=n_splines, spline_order=spline_order, lam=lam)

        # Fit the model
        final_gam_model.fit(X_train, y_train)

        # Predict on the test set
        y_test_hat_gam = final_gam_model.predict(X_test)
        
        # Calculate the accuracy
        accuracy_gam = accuracy_score(y_test, y_test_hat_gam)
    print("Accuracy GAM: ", accuracy_gam)

    # Load the existing DataFrame
    df = pd.read_csv(f'RESULTS/RANDOM/{task_id}_random_accuracy_results.csv')

    # Add the columns with accuracy of GAM and GP
    df.loc[df['Method'] == 'GAM', 'Accuracy'] = accuracy_gam

    # Create the directory if it doesn't exist
    os.makedirs('RESULTS/RANDOM', exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f'RESULTS/RANDOM/{task_id}_random_accuracy_results.csv', index=False)