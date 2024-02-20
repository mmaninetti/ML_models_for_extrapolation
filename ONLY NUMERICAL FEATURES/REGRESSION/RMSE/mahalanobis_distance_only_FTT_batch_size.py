
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
from utils import EarlyStopping, train, train_trans, train_no_early_stopping, train_trans_no_early_stopping, train_GP


SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

task_id=361072
task = openml.tasks.get_task(task_id)  # download the OpenML task
dataset = task.get_dataset()

X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute)

# Set the random seed for reproducibility
N_TRIALS=100
N_SAMPLES=100
PATIENCE=40
N_EPOCHS=1000
GP_ITERATIONS=1000
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
    print("Using GPU yo")
    X_train__tensor = X_train__tensor.cuda()
    y_train__tensor = y_train__tensor.cuda()
    X_train_tensor = X_train_tensor.cuda()
    y_train_tensor = y_train_tensor.cuda()
    X_val_tensor = X_val_tensor.cuda()
    y_val_tensor = y_val_tensor.cuda()
    X_test_tensor = X_test_tensor.cuda()
    y_test_tensor = y_test_tensor.cuda()
else:
    print("Using CPU yo")
print(X_train__tensor.device)

# Create flattened versions of the data
y_val_np = y_val.values.flatten()
y_test_np = y_test.values.flatten()

def train_2_ft(model, criterion, loss_list, optimizer, n_epochs, train_loader, val_loader, early_stopping):
    for epoch in range(n_epochs):
        for batch_X, batch_y in train_loader:
            # Move batch to device
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X, None).reshape(-1,)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # validate the model 
        y_val_hat = model(X_val_tensor, None).reshape(-1,)
        val_loss = criterion(y_val_hat, y_val_tensor)

        # Validation
        with torch.no_grad():
            val_loss = 0
            num_batches = 0
            for batch_X, batch_y in val_loader:
                # Move batch to device
                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                    batch_y = batch_y.cuda()

                # Forward pass and calculate loss
                outputs = model(batch_X, None).reshape(-1,)
                batch_loss = criterion(outputs, batch_y)

                # Accumulate batch loss
                val_loss += batch_loss.item()
                num_batches += 1

            # Calculate average validation loss
            val_loss /= num_batches

            # Check if early stopping condition is met
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")

    return n_epochs


#### FFTransformer
from torch.utils.data import TensorDataset, DataLoader
d_out = 1  
d_in=X_train_.shape[1]

def FTTrans_opt(trial):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    n_blocks = trial.suggest_int("n_blocks", 1, 5)
    d_block_multiplier = trial.suggest_int("d_block_multiplier", 1, 25)
    attention_n_heads = trial.suggest_int("attention_n_heads", 1, 20)
    attention_dropout = trial.suggest_float("attention_dropout", 0, 1)
    ffn_d_hidden_multiplier=trial.suggest_float("ffn_d_hidden_multiplier", 0.5, 3)
    ffn_dropout = trial.suggest_float("ffn_dropout", 0, 1)
    residual_dropout = trial.suggest_float("residual_dropout", 0, 1)

    FTTrans_model = FTTransformer(
    n_cont_features=d_in,
    cat_cardinalities=[],
    d_out=d_out,
    n_blocks=n_blocks,
    d_block=d_block_multiplier*attention_n_heads,
    attention_n_heads=attention_n_heads,
    attention_dropout=attention_dropout,
    ffn_d_hidden=None,
    ffn_d_hidden_multiplier=ffn_d_hidden_multiplier,
    ffn_dropout=ffn_dropout,
    residual_dropout=residual_dropout,
    )

    # Define your batch size
    batch_size = 256

    # Create TensorDatasets for training and validation sets
    train_dataset = TensorDataset(X_train__tensor, y_train__tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        FTTrans_model = FTTrans_model.cuda()

    n_epochs=N_EPOCHS
    learning_rate=trial.suggest_float('learning_rate', 0.0001, 0.05, log=True)
    weight_decay=trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
    optimizer=torch.optim.Adam(FTTrans_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    loss_Adam=[]

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=False)
    n_epochs=train_2_ft(FTTrans_model, criterion, loss_Adam, optimizer, n_epochs, train_loader, val_loader, early_stopping)
    n_epochs = trial.suggest_int('n_epochs', n_epochs, n_epochs)

    # Point prediction
    
    predictions = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_predictions = FTTrans_model(batch_X, None).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_val_hat_FTTrans = np.concatenate(predictions)
    RMSE_FTTrans=torch.sqrt(torch.mean(torch.square(y_val_tensor - y_val_hat_FTTrans)))

    return RMSE_FTTrans

sampler_FTTrans = optuna.samplers.TPESampler(seed=seed)
study_FTTrans = optuna.create_study(sampler=sampler_FTTrans, direction='minimize')
study_FTTrans.optimize(FTTrans_opt, n_trials=N_TRIALS)


FTTrans_model = FTTransformer(
    n_cont_features=d_in,
    cat_cardinalities=[],
    d_out=d_out,
    n_blocks=study_FTTrans.best_params['n_blocks'],
    d_block=study_FTTrans.best_params['d_block_multiplier']*study_FTTrans.best_params['attention_n_heads'],
    attention_n_heads=study_FTTrans.best_params['attention_n_heads'],
    attention_dropout=study_FTTrans.best_params['attention_dropout'],
    ffn_d_hidden=None,
    ffn_d_hidden_multiplier=study_FTTrans.best_params['ffn_d_hidden_multiplier'],
    ffn_dropout=study_FTTrans.best_params['ffn_dropout'],
    residual_dropout=study_FTTrans.best_params['residual_dropout'],
    )

if torch.cuda.is_available():
    FTTrans_model = FTTrans_model.cuda()

n_epochs=study_FTTrans.best_params['n_epochs'] 
learning_rate=study_FTTrans.best_params['learning_rate']
weight_decay=study_FTTrans.best_params['weight_decay']
optimizer=torch.optim.Adam(FTTrans_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.MSELoss()
loss_Adam=[]

train_trans_no_early_stopping(FTTrans_model, criterion, loss_Adam, optimizer, n_epochs, X_train_tensor, y_train_tensor)

# Point prediction
y_test_hat_FTTrans = (FTTrans_model(X_test_tensor, None).reshape(-1,))
RMSE_FTTrans=torch.sqrt(torch.mean(torch.square(y_test_tensor - y_test_hat_FTTrans)))
print("RMSE FTTrans: ", RMSE_FTTrans)