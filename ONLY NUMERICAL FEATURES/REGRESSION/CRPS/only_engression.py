import os
import pandas as pd
import numpy as np
import setuptools
import openml
from sklearn.linear_model import LinearRegression 
import lightgbm as lgbm
import lightgbmlss
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
from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *
from pygam import LinearGAM, s, f
from utils import EarlyStopping, train, train_trans, train_no_early_stopping, train_trans_no_early_stopping, train_GP, ExactGPModel
from torch.utils.data import TensorDataset, DataLoader
from drf import drf

SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

task_id=361072

print(f"Task {task_id}")

# Create the checkpoint directory if it doesn't exist
os.makedirs('CHECKPOINTS/MAHALANOBIS', exist_ok=True)
CHECKPOINT_PATH = f'CHECKPOINTS/MAHALANOBIS/task_{task_id}.pt'

print(f"Task {task_id}")

task = openml.tasks.get_task(task_id)  # download the OpenML task
dataset = task.get_dataset()

X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute)

# Set the random seed for reproducibility
N_TRIALS=2
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

def engressor_NN(trial):

    params = {'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
            'num_epoches': trial.suggest_int('num_epoches', 100, 1000),
            'num_layer': trial.suggest_int('num_layer', 2, 5),
            'hidden_dim': trial.suggest_int('hidden_dim', 100, 500),
            'resblock': trial.suggest_categorical('resblock', [True, False])}
    params['noise_dim']=params['hidden_dim']

    # Check if CUDA is available and if so, move the tensors and the model to the GPU
    if torch.cuda.is_available():
        engressor_model=engression(X_train__tensor, y_train__tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=BATCH_SIZE, resblock=params['resblock'], device="cuda")
    else:
        engressor_model=engression(X_train__tensor, y_train__tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=BATCH_SIZE, resblock=params['resblock'])
    
    # Generate a sample from the engression model for each data point
    y_val_hat_engression_samples = [engressor_model.sample(torch.Tensor(np.array([X_val.values[i]])), sample_size=N_SAMPLES) for i in range(len(X_val))]

    # Calculate the CRPS for each prediction
    crps_values = [crps_ensemble(y_val_np[i], np.array(y_val_hat_engression_samples[i].cpu()).reshape(-1,)) for i in range(len(y_val_np))]

    return np.mean(crps_values.cpu())

sampler_engression = optuna.samplers.TPESampler(seed=seed)
study_engression = optuna.create_study(sampler=sampler_engression, direction='minimize')
study_engression.optimize(engressor_NN, n_trials=N_TRIALS)


params=study_engression.best_params
params['noise_dim']=params['hidden_dim']
X_train_tensor = torch.Tensor(np.array(X_train))
y_train_tensor = torch.Tensor(np.array(y_train).reshape(-1,1))

# Check if CUDA is available and if so, move the tensors and the model to the GPU
if torch.cuda.is_available():
    engressor_model=engression(X_train_tensor, y_train_tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=BATCH_SIZE, resblock=params['resblock'], device="cuda")
else:
    engressor_model=engression(X_train_tensor, y_train_tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=BATCH_SIZE, resblock=params['resblock'])
# Generate a sample from the engression model for each data point
y_test_hat_engression_samples = [engressor_model.sample(torch.Tensor(np.array([X_test.values[i]])).cuda() if torch.cuda.is_available() else torch.Tensor(np.array([X_test.values[i]])), sample_size=N_SAMPLES) for i in range(len(X_test))]
# Calculate the CRPS for each prediction
crps_values = [crps_ensemble(y_test_np[i], np.array(y_test_hat_engression_samples[i].cpu()).reshape(-1,)) for i in range(len(y_test_np))]
CRPS_engression=np.mean(crps_values.cpu())

print("CRPS engression", CRPS_engression)