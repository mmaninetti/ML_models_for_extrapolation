import pandas as pd
import numpy as np
import setuptools
import openml
from sklearn.linear_model import LinearRegression 
import lightgbm as lgbm
import optuna
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern
from engression import engression, engression_bagged
import torch
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
import random
import gpytorch
import tqdm.auto as tqdm
import os
from pygam import LinearGAM, s, f
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd

SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

# Specify the directory path
directory = 'RESULTS/CLUSTERING'

# Get all the file names in the directory
file_names = os.listdir(directory)

# Extract the task IDs from the file names
task_ids = [int(file_name.split('_')[0]) for file_name in file_names]

for task_id in task_ids:

    print(f"Task {task_id}")

    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()

    X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute)
    
    if len(X)>=100000:
        continue
    
    # Find features with absolute correlation > 0.9
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

    # Drop one of the highly correlated features
    X = X.drop(high_corr_features, axis=1)

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


    # New new implementation
    N_CLUSTERS=20
    # calculate the mean and covariance matrix of the dataset
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    scaler = StandardScaler()

    # transform data to compute the clusters
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init="auto").fit(X_scaled)
    distances=[]
    mahalanobis_dist=[]
    counts=[]
    ideal_len=len(kmeans.labels_)/5
    for i in np.arange(N_CLUSTERS):
        distances.append(np.abs(np.sum(kmeans.labels_==i)-ideal_len))
        counts.append(np.sum(kmeans.labels_==i))
        mean_k= np.mean(X.loc[kmeans.labels_==i,:], axis=0)
        mahalanobis_dist.append(mahalanobis(mean_k, mean, np.linalg.inv(cov)))

    dist_df=pd.DataFrame(data={'mahalanobis_dist': mahalanobis_dist, 'count': counts}, index=np.arange(N_CLUSTERS))
    dist_df=dist_df.sort_values('mahalanobis_dist', ascending=False)
    dist_df['cumulative_count']=dist_df['count'].cumsum()
    dist_df['abs_diff']=np.abs(dist_df['cumulative_count']-ideal_len)

    final=(np.where(dist_df['abs_diff']==np.min(dist_df['abs_diff']))[0])[0]
    labelss=dist_df.index[0:final+1].to_list()
    labels=pd.Series(kmeans.labels_).isin(labelss)
    labels.index=X.index
    close_index=labels.index[np.where(labels==False)[0]]
    far_index=labels.index[np.where(labels==True)[0]]

    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    # calculate the mean and covariance matrix of the dataset
    mean_ = np.mean(X_train, axis=0)
    cov_ = np.cov(X_train.T)
    scaler_ = StandardScaler()

    # transform data to compute the clusters
    X_train_scaled = scaler_.fit_transform(X_train)

    kmeans_ = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init="auto").fit(X_train_scaled)
    distances_=[]
    counts_=[]
    mahalanobis_dist_=[]
    ideal_len_=len(kmeans_.labels_)/5
    for i in np.arange(N_CLUSTERS):
        distances_.append(np.abs(np.sum(kmeans_.labels_==i)-ideal_len_))
        counts_.append(np.sum(kmeans_.labels_==i))
        mean_k_= np.mean(X_train.loc[kmeans_.labels_==i,:], axis=0)
        mahalanobis_dist_.append(mahalanobis(mean_k_, mean_, np.linalg.inv(cov_)))

    dist_df_=pd.DataFrame(data={'mahalanobis_dist': mahalanobis_dist_, 'count': counts_}, index=np.arange(N_CLUSTERS))
    dist_df_=dist_df_.sort_values('mahalanobis_dist', ascending=False)
    dist_df_['cumulative_count']=dist_df_['count'].cumsum()
    dist_df_['abs_diff']=np.abs(dist_df_['cumulative_count']-ideal_len_)

    final_=(np.where(dist_df_['abs_diff']==np.min(dist_df_['abs_diff']))[0])[0]
    labelss_=dist_df_.index[0:final_+1].to_list()
    labels_=pd.Series(kmeans_.labels_).isin(labelss_)
    labels_.index=X_train.index
    close_index_=labels_.index[np.where(labels_==False)[0]]
    far_index_=labels_.index[np.where(labels_==True)[0]]

    X_train_ = X_train.loc[close_index_,:]
    X_val = X_train.loc[far_index_,:]
    y_train_ = y_train.loc[close_index_]
    y_val = y_train.loc[far_index_]


    # Standardize the data
    mean_X_train_ = np.mean(X_train_, axis=0)
    std_X_train_ = np.std(X_train_, axis=0)
    X_train__scaled = (X_train_ - mean_X_train_) / std_X_train_
    X_val_scaled = (X_val - mean_X_train_) / std_X_train_

    mean_X_train = np.mean(X_train, axis=0)
    std_X_train = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean_X_train) / std_X_train
    X_test_scaled = (X_test - mean_X_train) / std_X_train


    # Convert data to PyTorch tensors
    X_train__tensor = torch.tensor(X_train__scaled.values, dtype=torch.float32)
    y_train__tensor = torch.tensor(y_train_.values, dtype=torch.float32)
    X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
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

    constant_prediction = np.array([np.mean(y_train)]*len(y_test))
    RMSE_constant = np.sqrt(np.mean((y_test - constant_prediction) ** 2))

    # Load the clustering results
    results = pd.read_csv('RESULTS/CLUSTERING/'+str(task_id)+'_clustering_RMSE_results.csv')

    # Update the value of the "constant" row with RMSE_constant
    results['RMSE'][0] = RMSE_constant

    # Save the updated clustering results
    results.to_csv('RESULTS/CLUSTERING/'+str(task_id)+'_clustering_RMSE_results.csv', index=False)


from umap import UMAP
import pandas as pd
import numpy as np
import setuptools
import openml
from sklearn.linear_model import LinearRegression 
import lightgbm as lgbm
import optuna
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern
from engression import engression, engression_bagged
import torch
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
from properscoring import crps_gaussian, crps_ensemble
import random
import gpytorch
import tqdm.auto as tqdm
from sklearn.metrics.pairwise import euclidean_distances
import os
from pygam import LinearGAM, s, f
from torch.utils.data import TensorDataset, DataLoader

SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

# Specify the directory path
directory = 'RESULTS/UMAP_DECOMPOSITION'

# Get all the file names in the directory
file_names = os.listdir(directory)

# Extract the task IDs from the file names
task_ids = [int(file_name.split('_')[0]) for file_name in file_names]

for task_id in task_ids:

    print(f"Task {task_id}")

    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()

    X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute)
    
    if len(X)>=100000:
        continue
    
    # Find features with absolute correlation > 0.9
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

    # Drop one of the highly correlated features
    X = X.drop(high_corr_features, axis=1)

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


    # Apply UMAP decomposition
    umap = UMAP(n_components=2, random_state=42)
    X_umap = umap.fit_transform(X)

    # calculate the Euclidean distance matrix
    euclidean_dist_matrix = euclidean_distances(X_umap)

    # calculate the Euclidean distance for each data point
    euclidean_dist = np.mean(euclidean_dist_matrix, axis=1)

    euclidean_dist = pd.Series(euclidean_dist, index=X.index)
    far_index = euclidean_dist.index[np.where(euclidean_dist >= np.quantile(euclidean_dist, 0.8))[0]]
    close_index = euclidean_dist.index[np.where(euclidean_dist < np.quantile(euclidean_dist, 0.8))[0]]

    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    # Apply UMAP decomposition on the training set
    X_umap_train = umap.fit_transform(X_train)

    # calculate the Euclidean distance matrix for the training set
    euclidean_dist_matrix_train = euclidean_distances(X_umap_train)

    # calculate the Euclidean distance for each data point in the training set
    euclidean_dist_train = np.mean(euclidean_dist_matrix_train, axis=1)

    euclidean_dist_train = pd.Series(euclidean_dist_train, index=X_train.index)
    far_index_train = euclidean_dist_train.index[np.where(euclidean_dist_train >= np.quantile(euclidean_dist_train, 0.8))[0]]
    close_index_train = euclidean_dist_train.index[np.where(euclidean_dist_train < np.quantile(euclidean_dist_train, 0.8))[0]]

    X_train_ = X_train.loc[close_index_train,:]
    X_val = X_train.loc[far_index_train,:]
    y_train_ = y_train.loc[close_index_train]
    y_val = y_train.loc[far_index_train]


        # Standardize the data
    mean_X_train_ = np.mean(X_train_, axis=0)
    std_X_train_ = np.std(X_train_, axis=0)
    X_train__scaled = (X_train_ - mean_X_train_) / std_X_train_
    X_val_scaled = (X_val - mean_X_train_) / std_X_train_

    mean_X_train = np.mean(X_train, axis=0)
    std_X_train = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean_X_train) / std_X_train
    X_test_scaled = (X_test - mean_X_train) / std_X_train


    # Convert data to PyTorch tensors
    X_train__tensor = torch.tensor(X_train__scaled.values, dtype=torch.float32)
    y_train__tensor = torch.tensor(y_train_.values, dtype=torch.float32)
    X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
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

    constant_prediction = np.array([np.mean(y_train)]*len(y_test))
    RMSE_constant = np.sqrt(np.mean((y_test - constant_prediction) ** 2))

    # Load the clustering results
    results = pd.read_csv('RESULTS/UMAP_DECOMPOSITION/'+str(task_id)+'_umap_decomposition_RMSE_results.csv')

    # Update the value of the "constant" row with RMSE_constant
    results['RMSE'][0] = RMSE_constant

    # Save the updated clustering results
    results.to_csv('RESULTS/UMAP_DECOMPOSITION/'+str(task_id)+'_umap_decomposition_RMSE_results.csv', index=False)


import pandas as pd
import numpy as np
import setuptools
import openml
from sklearn.linear_model import LinearRegression 
import lightgbm as lgbm
import optuna
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern
from engression import engression, engression_bagged
import torch
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
from properscoring import crps_gaussian, crps_ensemble
import random
import gpytorch
import tqdm.auto as tqdm
import os
from pygam import LinearGAM, s, f
from torch.utils.data import TensorDataset, DataLoader

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

# Specify the directory path
directory = 'RESULTS/SPATIAL_DEPTH'

# Get all the file names in the directory
file_names = os.listdir(directory)

# Extract the task IDs from the file names
task_ids = [int(file_name.split('_')[0]) for file_name in file_names]

for task_id in task_ids:

    print(f"Task {task_id}")

    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()

    X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute)
    
    if len(X)>=100000:
        continue
    
    # Find features with absolute correlation > 0.9
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

    # Drop one of the highly correlated features
    X = X.drop(high_corr_features, axis=1)

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


    # activate pandas conversion for rpy2
    pandas2ri.activate()

    # import R's "ddalpha" package
    ddalpha = importr('ddalpha')

    # explicitly import the projDepth function
    spatialDepth = robjects.r['depth.spatial']

    # calculate the spatial depth for each data point
    spatial_depth = spatialDepth(X, X)

    spatial_depth=pd.Series(spatial_depth,index=X.index)
    far_index=spatial_depth.index[np.where(spatial_depth<=np.quantile(spatial_depth,0.2))[0]]
    close_index=spatial_depth.index[np.where(spatial_depth>np.quantile(spatial_depth,0.2))[0]]

    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    # convert the R vector to a pandas Series
    spatial_depth_ = spatialDepth(X_train, X_train)

    spatial_depth_=pd.Series(spatial_depth_,index=X_train.index)
    far_index_=spatial_depth_.index[np.where(spatial_depth_<=np.quantile(spatial_depth_,0.2))[0]]
    close_index_=spatial_depth_.index[np.where(spatial_depth_>np.quantile(spatial_depth_,0.2))[0]]

    X_train_ = X_train.loc[close_index_,:]
    X_val = X_train.loc[far_index_,:]
    y_train_ = y_train.loc[close_index_]
    y_val = y_train.loc[far_index_]


        # Standardize the data
    mean_X_train_ = np.mean(X_train_, axis=0)
    std_X_train_ = np.std(X_train_, axis=0)
    X_train__scaled = (X_train_ - mean_X_train_) / std_X_train_
    X_val_scaled = (X_val - mean_X_train_) / std_X_train_

    mean_X_train = np.mean(X_train, axis=0)
    std_X_train = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean_X_train) / std_X_train
    X_test_scaled = (X_test - mean_X_train) / std_X_train


    # Convert data to PyTorch tensors
    X_train__tensor = torch.tensor(X_train__scaled.values, dtype=torch.float32)
    y_train__tensor = torch.tensor(y_train_.values, dtype=torch.float32)
    X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
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

    constant_prediction = np.array([np.mean(y_train)]*len(y_test))
    RMSE_constant = np.sqrt(np.mean((y_test - constant_prediction) ** 2))

    # Load the clustering results
    results = pd.read_csv('RESULTS/SPATIAL_DEPTH/'+str(task_id)+'_spatial_depth_RMSE_results.csv')

    # Update the value of the "constant" row with RMSE_constant
    results['RMSE'][0] = RMSE_constant

    # Save the updated clustering results
    results.to_csv('RESULTS/SPATIAL_DEPTH/'+str(task_id)+'_spatial_depth_RMSE_results.csv', index=False)


import pandas as pd
import numpy as np
import setuptools
import openml
from sklearn.linear_model import LinearRegression 
import lightgbm as lgbm
import optuna
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern
from engression import engression, engression_bagged
import torch
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
from properscoring import crps_gaussian, crps_ensemble
import random
import gpytorch
import tqdm.auto as tqdm
import os
from pygam import LinearGAM, s, f
from torch.utils.data import TensorDataset, DataLoader

SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

# Specify the directory path
directory = 'RESULTS/MAHALANOBIS'

# Get all the file names in the directory
file_names = os.listdir(directory)

# Extract the task IDs from the file names
task_ids = [int(file_name.split('_')[0]) for file_name in file_names]

for task_id in task_ids:

    print(f"Task {task_id}")

    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()

    X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute)
    
    if len(X)>=100000:
        continue
    
    # Find features with absolute correlation > 0.9
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

    # Drop one of the highly correlated features
    X = X.drop(high_corr_features, axis=1)

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


    # calculate the mean and covariance matrix of the dataset
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)

    # calculate the Mahalanobis distance for each data point
    mahalanobis_dist = [mahalanobis(x, mean, np.linalg.inv(cov)) for x in X.values]

    mahalanobis_dist=pd.Series(mahalanobis_dist,index=X.index)
    far_index=mahalanobis_dist.index[np.where(mahalanobis_dist>=np.quantile(mahalanobis_dist,0.8))[0]]
    close_index=mahalanobis_dist.index[np.where(mahalanobis_dist<np.quantile(mahalanobis_dist,0.8))[0]]

    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    mean = np.mean(X_train, axis=0)
    cov = np.cov(X_train.T)

    # calculate the Mahalanobis distance for each data point
    mahalanobis_dist_ = [mahalanobis(x, mean, np.linalg.inv(cov)) for x in X_train.values]

    mahalanobis_dist_=pd.Series(mahalanobis_dist_,index=X_train.index)
    far_index_=mahalanobis_dist_.index[np.where(mahalanobis_dist_>=np.quantile(mahalanobis_dist_,0.8))[0]]
    close_index_=mahalanobis_dist_.index[np.where(mahalanobis_dist_<np.quantile(mahalanobis_dist_,0.8))[0]]

    X_train_ = X_train.loc[close_index_,:]
    X_val = X_train.loc[far_index_,:]
    y_train_ = y_train.loc[close_index_]
    y_val = y_train.loc[far_index_]

    # Standardize the data
    mean_X_train_ = np.mean(X_train_, axis=0)
    std_X_train_ = np.std(X_train_, axis=0)
    X_train__scaled = (X_train_ - mean_X_train_) / std_X_train_
    X_val_scaled = (X_val - mean_X_train_) / std_X_train_

    mean_X_train = np.mean(X_train, axis=0)
    std_X_train = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean_X_train) / std_X_train
    X_test_scaled = (X_test - mean_X_train) / std_X_train


    # Convert data to PyTorch tensors
    X_train__tensor = torch.tensor(X_train__scaled.values, dtype=torch.float32)
    y_train__tensor = torch.tensor(y_train_.values, dtype=torch.float32)
    X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
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

    constant_prediction = np.array([np.mean(y_train)]*len(y_test))
    RMSE_constant = np.sqrt(np.mean((y_test - constant_prediction) ** 2))

    # Load the clustering results
    results = pd.read_csv('RESULTS/MAHALANOBIS/'+str(task_id)+'_mahalanobis_RMSE_results.csv')

    # Update the value of the "constant" row with RMSE_constant
    results['RMSE'][0] = RMSE_constant

    # Save the updated clustering results
    results.to_csv('RESULTS/MAHALANOBIS/'+str(task_id)+'_mahalanobis_RMSE_results.csv', index=False)