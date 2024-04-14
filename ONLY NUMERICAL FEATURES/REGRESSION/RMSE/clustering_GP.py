import pandas as pd
import numpy as np
import openml
from sklearn.linear_model import LinearRegression 
import lightgbm as lgbm
import optuna
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from engression import engression
import torch
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
import random
import os
from pygam import LinearGAM
from utils import EarlyStopping, train, train_trans, train_no_early_stopping, train_trans_no_early_stopping
from torch.utils.data import TensorDataset, DataLoader
import re
import shutil
import gpboost as gpb

SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

#task_id=361072
for task_id in benchmark_suite.tasks:

    # Set the random seed for reproducibility
    N_TRIALS=100
    N_SAMPLES=100
    PATIENCE=40
    N_EPOCHS=1000
    GP_ITERATIONS=1000
    BATCH_SIZE=1024
    seed=10
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    print(f"Task {task_id}")

    CHECKPOINT_PATH = f'CHECKPOINTS/CLUSTERING/task_{task_id}.pt'

    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()

    X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute)
    
    if (task_id==361082) or (task_id==361088):
        y=np.log(y)
    
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

    # New new implementation
    N_CLUSTERS=20
    # calculate the mean and covariance matrix of the dataset
    mean = np.mean(X_clean, axis=0)
    cov = np.cov(X_clean.T)
    scaler = StandardScaler()

    # transform data to compute the clusters
    X_clean_scaled = scaler.fit_transform(X_clean)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init="auto").fit(X_clean_scaled)
    distances=[]
    mahalanobis_dist=[]
    counts=[]
    ideal_len=len(kmeans.labels_)/5
    for i in np.arange(N_CLUSTERS):
        distances.append(np.abs(np.sum(kmeans.labels_==i)-ideal_len))
        counts.append(np.sum(kmeans.labels_==i))
        mean_k= np.mean(X_clean.loc[kmeans.labels_==i,:], axis=0)
        mahalanobis_dist.append(mahalanobis(mean_k, mean, np.linalg.inv(cov)))

    dist_df=pd.DataFrame(data={'mahalanobis_dist': mahalanobis_dist, 'count': counts}, index=np.arange(N_CLUSTERS))
    dist_df=dist_df.sort_values('mahalanobis_dist', ascending=False)
    dist_df['cumulative_count']=dist_df['count'].cumsum()
    dist_df['abs_diff']=np.abs(dist_df['cumulative_count']-ideal_len)

    final=(np.where(dist_df['abs_diff']==np.min(dist_df['abs_diff']))[0])[0]
    labelss=dist_df.index[0:final+1].to_list()
    labels=pd.Series(kmeans.labels_).isin(labelss)
    labels.index=X_clean.index
    close_index=labels.index[np.where(labels==False)[0]]
    far_index=labels.index[np.where(labels==True)[0]]

    X_train_clean = X_clean.loc[close_index,:]
    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    # calculate the mean and covariance matrix of the dataset
    mean_ = np.mean(X_train_clean, axis=0)
    cov_ = np.cov(X_train_clean.T)
    scaler_ = StandardScaler()

    # transform data to compute the clusters
    X_train_clean_scaled = scaler_.fit_transform(X_train_clean)

    kmeans_ = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init="auto").fit(X_train_clean_scaled)
    distances_=[]
    counts_=[]
    mahalanobis_dist_=[]
    ideal_len_=len(kmeans_.labels_)/5
    for i in np.arange(N_CLUSTERS):
        distances_.append(np.abs(np.sum(kmeans_.labels_==i)-ideal_len_))
        counts_.append(np.sum(kmeans_.labels_==i))
        mean_k_= np.mean(X_train_clean.loc[kmeans_.labels_==i,:], axis=0)
        mahalanobis_dist_.append(mahalanobis(mean_k_, mean_, np.linalg.inv(cov_)))

    dist_df_=pd.DataFrame(data={'mahalanobis_dist': mahalanobis_dist_, 'count': counts_}, index=np.arange(N_CLUSTERS))
    dist_df_=dist_df_.sort_values('mahalanobis_dist', ascending=False)
    dist_df_['cumulative_count']=dist_df_['count'].cumsum()
    dist_df_['abs_diff']=np.abs(dist_df_['cumulative_count']-ideal_len_)

    final_=(np.where(dist_df_['abs_diff']==np.min(dist_df_['abs_diff']))[0])[0]
    labelss_=dist_df_.index[0:final_+1].to_list()
    labels_=pd.Series(kmeans_.labels_).isin(labelss_)
    labels_.index=X_train_clean.index
    close_index_=labels_.index[np.where(labels_==False)[0]]
    far_index_=labels_.index[np.where(labels_==True)[0]]

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
        print("Using GPU")
        X_train__tensor = X_train__tensor.cuda()
        y_train__tensor = y_train__tensor.cuda()
        X_train_tensor = X_train_tensor.cuda()
        y_train_tensor = y_train_tensor.cuda()
        X_val_tensor = X_val_tensor.cuda()
        y_val_tensor = y_val_tensor.cuda()
        X_test_tensor = X_test_tensor.cuda()
        y_test_tensor = y_test_tensor.cuda()
    else:
        print("Using CPU")

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

    # Define d_out and d_in
    d_out = 1  
    d_in=X_train_.shape[1]

    #### GP model
    approximations = ["vecchia", "fitc"]
    kernels = ["matern_ard", "gaussian_ard"]
    shapes = [0.5, 1.5, 2.5]
    best_RMSE = float('inf')    
    intercept_train=np.ones(X_train_.shape[0])
    intercept_val=np.ones(X_val.shape[0])
    for approx in approximations:
        for kernel in kernels:
            if kernel=="matern_ard":
                for shape in shapes:
                    gp_model = gpb.GPModel(gp_coords=X_train_, cov_function=kernel, cov_fct_shape=shape, likelihood="gaussian", gp_approx=approx, seed=seed)
                    gp_model.fit(y=y_train_, X=intercept_train, params={"trace": True})
                    pred_resp = gp_model.predict(gp_coords_pred=X_val, X_pred=intercept_val, predict_var=True, predict_response=True)['mu']
                    RMSE_GP = np.sqrt(np.mean((y_val-pred_resp)**2))
                    print("RMSE GP temporary: ", RMSE_GP)
                    if RMSE_GP < best_RMSE:
                        best_RMSE = RMSE_GP
                        best_approx = approx
                        best_kernel = kernel
                        best_shape = shape
            else:
                gp_model = gpb.GPModel(gp_coords=X_train_, cov_function=kernel, likelihood="gaussian", gp_approx=approx, seed=seed)
                gp_model.fit(y=y_train_, X=intercept_train, params={"trace": True})
                pred_resp = gp_model.predict(gp_coords_pred=X_val, X_pred=intercept_val, predict_var=True, predict_response=True)['mu']
                RMSE_GP = np.sqrt(np.mean((y_val-pred_resp)**2))
                print("RMSE GP temporary: ", RMSE_GP)
                if RMSE_GP < best_RMSE:
                    best_RMSE = RMSE_GP
                    best_approx = approx
                    best_kernel = kernel
                    best_shape = None
    
    intercept_train=np.ones(X_train.shape[0])
    intercept_test=np.ones(X_test.shape[0])
    if best_kernel=="matern_ard":
        gp_model = gpb.GPModel(gp_coords=X_train, cov_function=best_kernel, cov_fct_shape=best_shape, likelihood="gaussian", gp_approx=best_approx, seed=seed)
    else:
        gp_model = gpb.GPModel(gp_coords=X_train, cov_function=best_kernel, likelihood="gaussian", gp_approx=best_approx, seed=seed)
    
    gp_model.fit(y=y_train, X=intercept_train, params={"trace": True})
    pred_resp = gp_model.predict(gp_coords_pred=X_test, X_pred=intercept_test, predict_var=True, predict_response=True)['mu']
    RMSE_GP = np.sqrt(np.mean((y_test-pred_resp)**2))    
    print("RMSE GP: ", RMSE_GP)

    # Load the existing DataFrame
    df = pd.read_csv(f'RESULTS/CLUSTERING/{task_id}_clustering_RMSE_results.csv')

    # Add the columns with RMSE of GP and GP
    if 'GP' in df['Method'].values:
        df.loc[df['Method'] == 'GP', 'RMSE'] = RMSE_GP
    else:
        df.loc[len(df)] = ['GP', RMSE_GP]

    # Create the directory if it doesn't exist
    os.makedirs('RESULTS/CLUSTERING', exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f'RESULTS/CLUSTERING/{task_id}_clustering_RMSE_results.csv', index=False)