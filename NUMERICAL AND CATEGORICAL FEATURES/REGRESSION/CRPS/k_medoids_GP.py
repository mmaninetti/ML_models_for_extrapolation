import os
import pandas as pd
import numpy as np
import openml
from sklearn.linear_model import LinearRegression 
import lightgbm as lgbm
import optuna
from sklearn.ensemble import RandomForestRegressor
from engression import engression
import torch
from scipy.stats import norm
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
from properscoring import crps_gaussian, crps_ensemble
import random
from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *
from drf import drf
from pygam import LinearGAM
import gower
from sklearn_extra.cluster import KMedoids
from utils import EarlyStopping, train, train_trans, train_no_early_stopping, train_trans_no_early_stopping
from torch.utils.data import TensorDataset, DataLoader
import shutil
import gpboost as gpb

# Create the checkpoint directory if it doesn't exist
if os.path.exists('CHECKPOINTS/KMEDOIDS'):
    shutil.rmtree('CHECKPOINTS/KMEDOIDS')
os.makedirs('CHECKPOINTS/KMEDOIDS')

#openml.config.apikey = 'FILL_IN_OPENML_API_KEY'  # set the OpenML Api Key
#SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

#task_id=361093
for task_id in benchmark_suite.tasks:

    if task_id != 361093 and task_id<361289:
        continue

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

    CHECKPOINT_PATH = f'CHECKPOINTS/KMEDOIDS/task_{task_id}.pt'

    print(f"Task {task_id}")

    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()

    X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute)
    
    if task_id==361099:
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


    N_CLUSTERS=20

    for col in X_clean.select_dtypes(['category']).columns:
        X_clean[col] = X_clean[col].astype('object')

    gower_dist_matrix = gower.gower_matrix(X_clean)

    kmedoids = KMedoids(n_clusters=N_CLUSTERS, random_state=0, metric='precomputed', init='k-medoids++').fit(gower_dist_matrix)
    distances=[]
    gower_dist=[]
    counts=[]
    ideal_len=len(kmedoids.labels_)/5

    for i in np.arange(N_CLUSTERS):
        cluster_data = X_clean.loc[kmedoids.labels_==i,:]
        # Compute the Gower distance between each data point in the cluster and each data point in the global dataset
        distances_matrix = gower.gower_matrix(cluster_data, X_clean)
        # Compute the average distance
        average_distance = np.mean(distances_matrix)
        gower_dist.append(average_distance)
        counts.append(cluster_data.shape[0])

    dist_df=pd.DataFrame(data={'gower_dist': gower_dist, 'count': counts}, index=np.arange(N_CLUSTERS))
    dist_df=dist_df.sort_values('gower_dist', ascending=False)
    dist_df['cumulative_count']=dist_df['count'].cumsum()
    dist_df['abs_diff']=np.abs(dist_df['cumulative_count']-ideal_len)

    final=(np.where(dist_df['abs_diff']==np.min(dist_df['abs_diff']))[0])[0]
    labelss=dist_df.index[0:final+1].to_list()
    labels=pd.Series(kmedoids.labels_).isin(labelss)
    labels.index=X_clean.index
    close_index=labels.index[np.where(labels==False)[0]]
    far_index=labels.index[np.where(labels==True)[0]]

    X_clean_ = X_clean.loc[close_index,:]

    for col in X_clean_.select_dtypes(['category']).columns:
        X_clean_[col] = X_clean_[col].astype('object')

    gower_dist_matrix_ = gower.gower_matrix(X_clean_)

    kmedoids_ = KMedoids(n_clusters=N_CLUSTERS, random_state=0, metric='precomputed', init='k-medoids++').fit(gower_dist_matrix_)
    distances_=[]
    gower_dist_=[]
    counts_=[]
    ideal_len_=len(kmedoids.labels_)/5

    for i in np.arange(N_CLUSTERS):
        cluster_data_ = X_clean_.loc[kmedoids_.labels_==i,:]
        # Compute the Gower distance between each data point in the cluster and each data point in the global dataset
        distances_matrix_ = gower.gower_matrix(cluster_data_, X_clean_)
        # Compute the average distance
        average_distance_ = np.mean(distances_matrix_)
        gower_dist_.append(average_distance_)
        counts_.append(cluster_data_.shape[0])

    dist_df_=pd.DataFrame(data={'gower_dist': gower_dist_, 'count': counts_}, index=np.arange(N_CLUSTERS))
    dist_df_=dist_df_.sort_values('gower_dist', ascending=False)
    dist_df_['cumulative_count']=dist_df_['count'].cumsum()
    dist_df_['abs_diff']=np.abs(dist_df_['cumulative_count']-ideal_len_)

    final_=(np.where(dist_df_['abs_diff']==np.min(dist_df_['abs_diff']))[0])[0]
    labelss_=dist_df_.index[0:final_+1].to_list()
    labels_=pd.Series(kmedoids_.labels_).isin(labelss_)
    labels_.index=X_clean_.index
    close_index_train=labels_.index[np.where(labels_==False)[0]]
    far_index_train=labels_.index[np.where(labels_==True)[0]]

    # Check if categorical variables have the same cardinality in X and X_train_, and remove the ones that don't
    dummy_cols = X.select_dtypes(['bool', 'category', 'object', 'string']).columns
    X_train = X.loc[close_index,:]
    X_train_ = X_train.loc[close_index_train,:]
    for col in dummy_cols:
        if len(X[col].unique()) != len(X_train_[col].unique()):
            X = X.drop(col, axis=1)

    # Convert data to PyTorch tensors
    # Modify X_train_, X_val, X_train, and X_test to have dummy variables
    non_dummy_cols = X.select_dtypes(exclude=['bool', 'category', 'object', 'string']).columns
    X = pd.get_dummies(X, drop_first=True).astype('float32')

    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    X_train_ = X_train.loc[close_index_train,:]
    X_val = X_train.loc[far_index_train,:]
    y_train_ = y_train.loc[close_index_train]
    y_val = y_train.loc[far_index_train]

    # Standardize the data for non-dummy variables
    mean_X_train_ = np.mean(X_train_[non_dummy_cols], axis=0)
    std_X_train_ = np.std(X_train_[non_dummy_cols], axis=0)
    X_train_[non_dummy_cols] = (X_train_[non_dummy_cols] - mean_X_train_) / std_X_train_
    X_val = X_val.copy()
    X_val[non_dummy_cols] = (X_val[non_dummy_cols] - mean_X_train_) / std_X_train_

    mean_X_train = np.mean(X_train[non_dummy_cols], axis=0)
    std_X_train = np.std(X_train[non_dummy_cols], axis=0)
    X_train[non_dummy_cols] = (X_train[non_dummy_cols] - mean_X_train) / std_X_train
    X_test = X_test.copy()
    X_test[non_dummy_cols] = (X_test[non_dummy_cols] - mean_X_train) / std_X_train

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

    # Define d_out and d_in
    d_out = 1  
    d_in=X_train_.shape[1]


    #### GP model
    approximations = ["vecchia", "fitc"]
    if task_id==361093 or task_id==361289:
        approximations = ["vecchia"]
    kernels = ["matern", "gaussian"]
    shapes = [0.5, 1.5, 2.5]
    best_CRPS = float('inf')    
    intercept_train=np.ones(X_train_.shape[0])
    intercept_val=np.ones(X_val.shape[0])
    for approx in approximations:
        for kernel in kernels:
            if kernel=="matern":
                for shape in shapes:
                    gp_model = gpb.GPModel(gp_coords=X_train_, cov_function=kernel, cov_fct_shape=shape, likelihood="gaussian", gp_approx=approx, seed=seed)
                    gp_model.fit(y=y_train_, X=intercept_train, params={"trace": True})
                    pred_mu = gp_model.predict(gp_coords_pred=X_val, X_pred=intercept_val, predict_var=True, predict_response=True)['mu']
                    pred_std = gp_model.predict(gp_coords_pred=X_val, X_pred=intercept_val, predict_var=True, predict_response=True)['var']
                    crps_values = [crps_gaussian(y_val_np[i], mu=pred_mu[i], sig=pred_std[i]) for i in range(len(y_val))] 
                    CRPS_GP = np.mean(crps_values)
                    print("CRPS GP temporary: ", CRPS_GP)
                    if CRPS_GP < best_CRPS and CRPS_GP > 0:
                        best_CRPS = CRPS_GP
                        best_approx = approx
                        best_kernel = kernel
                        best_shape = shape
            else:
                gp_model = gpb.GPModel(gp_coords=X_train_, cov_function=kernel, likelihood="gaussian", gp_approx=approx, seed=seed)
                gp_model.fit(y=y_train_, X=intercept_train, params={"trace": True})
                pred_mu = gp_model.predict(gp_coords_pred=X_val, X_pred=intercept_val, predict_var=True, predict_response=True)['mu']
                pred_std = gp_model.predict(gp_coords_pred=X_val, X_pred=intercept_val, predict_var=True, predict_response=True)['var']
                crps_values = [crps_gaussian(y_val_np[i], mu=pred_mu[i], sig=pred_std[i]) for i in range(len(y_val))] 
                CRPS_GP = np.mean(crps_values)
                print("CRPS GP temporary: ", CRPS_GP)
                if CRPS_GP < best_CRPS and CRPS_GP > 0:
                    best_CRPS = CRPS_GP
                    best_approx = approx
                    best_kernel = kernel
                    best_shape = None
    
    intercept_train=np.ones(X_train.shape[0])
    intercept_test=np.ones(X_test.shape[0])
    if best_kernel=="matern":
        gp_model = gpb.GPModel(gp_coords=X_train, cov_function=best_kernel, cov_fct_shape=best_shape, likelihood="gaussian", gp_approx=best_approx, seed=seed)
    else:
        gp_model = gpb.GPModel(gp_coords=X_train, cov_function=best_kernel, likelihood="gaussian", gp_approx=best_approx, seed=seed)
    
    gp_model.fit(y=y_train, X=intercept_train, params={"trace": True})
    pred_mu = gp_model.predict(gp_coords_pred=X_test, X_pred=intercept_test, predict_var=True, predict_response=True)['mu']
    pred_std = gp_model.predict(gp_coords_pred=X_test, X_pred=intercept_test, predict_var=True, predict_response=True)['var']
    crps_values = [crps_gaussian(y_test_np[i], mu=pred_mu[i], sig=pred_std[i]) for i in range(len(y_test))]  
    CRPS_GP = np.mean(crps_values)
    print("CRPS GP: ", CRPS_GP)

    # Load the existing DataFrame
    df = pd.read_csv(f'RESULTS/K_MEDOIDS/{task_id}_k_medoids_crps_results.csv')

    # Add the columns with CRPS of GP
    if 'GP' in df['Method'].values:
        df.loc[df['Method'] == 'GP', 'CRPS'] = CRPS_GP
    else:
        df.loc[len(df)] = ['GP', CRPS_GP]

    # Create the directory if it doesn't exist
    os.makedirs('RESULTS/K_MEDOIDS', exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f'RESULTS/K_MEDOIDS/{task_id}_k_medoids_crps_results.csv', index=False)