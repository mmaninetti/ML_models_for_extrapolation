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

    if task_id < 361292:
        continue

    if task_id == 361096 or task_id == 361097 or task_id == 361103 or task_id == 361287 or task_id == 361292:
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


    #### GAM model
    def gam_model(trial):

        # Define the search space for n_splines, lam, and spline_order
        n_splines=trial.suggest_int('n_splines', 10, 100)
        lam=trial.suggest_float('lam', 1e-3, 1e3, log=True)
        spline_order=trial.suggest_int('spline_order', 1, 5)
        
        # Create and train the model
        gam = LinearGAM(n_splines=n_splines, spline_order=spline_order, lam=lam).fit(X_train_, y_train_)

        # Predict on the validation set and calculate the CRPS
        y_train__hat_gam = gam.predict(X_train_)
        std_dev_error = np.std(y_train_ - y_train__hat_gam)
        y_val_hat_gam = gam.predict(X_val)
        crps_gam = [crps_gaussian(y_val_np[i], mu=y_val_hat_gam[i], sig=std_dev_error) for i in range(len(y_val_hat_gam))]
        crps_gam = np.mean(crps_gam)

        return crps_gam

    # Create the sampler and study
    sampler_gam = optuna.samplers.TPESampler(seed=seed)
    study_gam = optuna.create_study(sampler=sampler_gam, direction='minimize')

    # Optimize the model
    study_gam.optimize(gam_model, n_trials=N_TRIALS)

    # Get the best parameters
    best_params = study_gam.best_params
    n_splines=best_params['n_splines']
    lam=best_params['lam']
    spline_order=best_params['spline_order']

    final_gam_model = LinearGAM(n_splines=n_splines, spline_order=spline_order, lam=lam)

    # Fit the model
    final_gam_model.fit(X_train, y_train)

    # Predict on the train set
    y_train_hat_gam = final_gam_model.predict(X_train)
    std_dev_error = np.std(y_train - y_train_hat_gam)

    # Predict on the test set
    y_test_hat_gam = final_gam_model.predict(X_test)

    # Calculate the CRPS
    crps_gam = [crps_gaussian(y_test_np[i], mu=y_test_hat_gam[i], sig=std_dev_error) for i in range(len(y_test_hat_gam))]
    crps_gam = np.mean(crps_gam)
    print("CRPS GAM: ", crps_gam)

    # Load the existing DataFrame
    df = pd.read_csv(f'RESULTS/K_MEDOIDS/{task_id}_k_medoids_crps_results.csv')

    # Add the columns with CRPS of GAM and GP
    df.loc[df['Method'] == 'GAM', 'CRPS'] = crps_gam

    # Create the directory if it doesn't exist
    os.makedirs('RESULTS/K_MEDOIDS', exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f'RESULTS/K_MEDOIDS/{task_id}_k_medoids_crps_results.csv', index=False)