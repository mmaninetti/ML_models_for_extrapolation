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
from sklearn.preprocessing import LabelEncoder 
from utils import EarlyStopping, train, train_trans, train_no_early_stopping, train_trans_no_early_stopping
from torch.utils.data import TensorDataset, DataLoader
import re
import shutil
import gpboost as gpb

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

#SUITE_ID = 336 # Regression on numerical features
SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

#task_id=361055
for task_id in benchmark_suite.tasks:  # iterate over all tasks in the suite

    if task_id==361276:
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

    CHECKPOINT_PATH = f'CHECKPOINTS/SPATIAL_DEPTH/task_{task_id}.pt'

    print(f"Task {task_id}")

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


    # activate pandas conversion for rpy2
    pandas2ri.activate()

    # import R's "ddalpha" package
    ddalpha = importr('ddalpha')

    # explicitly import the projDepth function
    spatialDepth = robjects.r['depth.spatial']

    # calculate the spatial depth for each data point
    spatial_depth = spatialDepth(X_clean, X_clean)

    spatial_depth=pd.Series(spatial_depth,index=X_clean.index)
    far_index=spatial_depth.index[np.where(spatial_depth<=np.quantile(spatial_depth,0.2))[0]]
    close_index=spatial_depth.index[np.where(spatial_depth>np.quantile(spatial_depth,0.2))[0]]

    X_train_clean = X_clean.loc[close_index,:]
    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    # convert the R vector to a pandas Series
    spatial_depth_ = spatialDepth(X_train_clean, X_train_clean)

    spatial_depth_=pd.Series(spatial_depth_,index=X_train_clean.index)
    far_index_=spatial_depth_.index[np.where(spatial_depth_<=np.quantile(spatial_depth_,0.2))[0]]
    close_index_=spatial_depth_.index[np.where(spatial_depth_>np.quantile(spatial_depth_,0.2))[0]]

    X_train_ = X_train.loc[close_index_,:]
    X_val = X_train.loc[far_index_,:]
    y_train_ = y_train.loc[close_index_]
    y_val = y_train.loc[far_index_]


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

    #### GP model
    approximations = ["vecchia", "fitc"]
    kernels = ["matern", "gaussian"]
    shapes = [0.5, 1.5, 2.5]
    best_accuracy = 0  
    intercept_train=np.ones(X_train_.shape[0])
    intercept_val=np.ones(X_val.shape[0])
    for approx in approximations:
        for kernel in kernels:
            if kernel=="matern":
                for shape in shapes:
                    if approx=="vecchia":
                        gp_model = gpb.GPModel(gp_coords=X_train_, cov_function=kernel, cov_fct_shape=shape, likelihood="bernoulli_logit", gp_approx=approx, matrix_inversion_method="iterative", seed=seed)
                    else:
                        gp_model = gpb.GPModel(gp_coords=X_train_, cov_function=kernel, cov_fct_shape=shape, likelihood="bernoulli_logit", gp_approx=approx, seed=seed)
                    gp_model.fit(y=y_train_, X=intercept_train, params={"trace": True})
                    pred_resp = gp_model.predict(gp_coords_pred=X_val, X_pred=intercept_val, predict_var=False, predict_response=True)['mu']
                    pred_resp = np.where(pred_resp >= 0.5, 1, 0)
                    accuracy_GP = accuracy_score(y_val, pred_resp)
                    print("Accuracy GP temporary: ", accuracy_GP)
                    if accuracy_GP > best_accuracy:
                        best_accuracy = accuracy_GP
                        best_approx = approx
                        best_kernel = kernel
                        best_shape = shape
            else:
                if approx=="vecchia":
                    gp_model = gpb.GPModel(gp_coords=X_train_, cov_function=kernel, likelihood="bernoulli_logit", gp_approx=approx, matrix_inversion_method="iterative", seed=seed)
                else:
                    gp_model = gpb.GPModel(gp_coords=X_train_, cov_function=kernel, likelihood="bernoulli_logit", gp_approx=approx, seed=seed)
                gp_model.fit(y=y_train_, X=intercept_train, params={"trace": True})
                pred_resp = gp_model.predict(gp_coords_pred=X_val, X_pred=intercept_val, predict_var=False, predict_response=True)['mu']
                pred_resp = np.where(pred_resp >= 0.5, 1, 0)
                accuracy_GP = accuracy_score(y_val, pred_resp)
                print("Accuracy GP temporary: ", accuracy_GP)
                if accuracy_GP > best_accuracy:
                    best_accuracy = accuracy_GP
                    best_approx = approx
                    best_kernel = kernel
                    best_shape = None
    
    intercept_train=np.ones(X_train.shape[0])
    intercept_test=np.ones(X_test.shape[0])
    if best_kernel=="matern":
        if approx=="vecchia":
            gp_model = gpb.GPModel(gp_coords=X_train, cov_function=best_kernel, cov_fct_shape=best_shape, likelihood="bernoulli_logit", gp_approx=best_approx, matrix_inversion_method="iterative", seed=seed)
        else:
            gp_model = gpb.GPModel(gp_coords=X_train, cov_function=best_kernel, cov_fct_shape=best_shape, likelihood="bernoulli_logit", gp_approx=best_approx, seed=seed)
    else:
        if approx=="vecchia":
            gp_model = gpb.GPModel(gp_coords=X_train, cov_function=best_kernel, likelihood="bernoulli_logit", gp_approx=best_approx, matrix_inversion_method="iterative", seed=seed)
        else:
            gp_model = gpb.GPModel(gp_coords=X_train, cov_function=best_kernel, likelihood="bernoulli_logit", gp_approx=best_approx, seed=seed)

    gp_model.fit(y=y_train, X=intercept_train, params={"trace": True})
    pred_resp = gp_model.predict(gp_coords_pred=X_test, X_pred=intercept_test, predict_var=False, predict_response=True)['mu']
    pred_resp = np.where(pred_resp >= 0.5, 1, 0)
    accuracy_GP = accuracy_score(y_test, pred_resp)    
    print("accuracy GP: ", accuracy_GP)

    # Load the existing DataFrame
    df = pd.read_csv(f'RESULTS/SPATIAL_DEPTH/{task_id}_spatial_depth_accuracy_results.csv')

    # Update the DataFrame with the new results
    if 'GP' in df['Method'].values:
        df.loc[df['Method'] == 'GP', 'Accuracy'] = accuracy_GP
    else:
        df.loc[len(df)] = ['GP', accuracy_GP]

    # Create the directory if it doesn't exist
    os.makedirs('RESULTS/SPATIAL_DEPTH', exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f'RESULTS/SPATIAL_DEPTH/{task_id}_spatial_depth_accuracy_results.csv', index=False)