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
from drf import drf
from pygam import LinearGAM, s, f
from sklearn.preprocessing import StandardScaler
import gower
from sklearn_extra.cluster import KMedoids
from utils import EarlyStopping, train, train_trans, train_no_early_stopping, train_trans_no_early_stopping, train_GP, ExactGPModel
from torch.utils.data import TensorDataset, DataLoader

#openml.config.apikey = 'FILL_IN_OPENML_API_KEY'  # set the OpenML Api Key
#SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

#task_id=361093
for task_id in benchmark_suite.tasks:

    print(f"Task {task_id}")

    # Create the checkpoint directory if it doesn't exist
    os.makedirs('CHECKPOINTS/K_MEDOIDS', exist_ok=True)
    CHECKPOINT_PATH = f'CHECKPOINTS/K_MEDOIDS/task_{task_id}.pt'

    print(f"Task {task_id}")

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
    BATCH_SIZE=1024
    seed=10
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


    N_CLUSTERS=20

    X_gower = X.copy()

    for col in X_gower.select_dtypes(['category']).columns:
        X_gower[col] = X_gower[col].astype('object')

    gower_dist_matrix = gower.gower_matrix(X_gower)

    kmedoids = KMedoids(n_clusters=N_CLUSTERS, random_state=0, metric='precomputed', init='k-medoids++').fit(gower_dist_matrix)
    distances=[]
    gower_dist=[]
    counts=[]
    ideal_len=len(kmedoids.labels_)/5

    for i in np.arange(N_CLUSTERS):
        cluster_data = X_gower.loc[kmedoids.labels_==i,:]
        # Compute the Gower distance between each data point in the cluster and each data point in the global dataset
        distances_matrix = gower.gower_matrix(cluster_data, X_gower)
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
    labels.index=X.index
    close_index=labels.index[np.where(labels==False)[0]]
    far_index=labels.index[np.where(labels==True)[0]]

    X_train = X.loc[close_index,:]
    X_gower_ = X_train.copy()

    for col in X_gower_.select_dtypes(['category']).columns:
        X_gower_[col] = X_gower_[col].astype('object')

    gower_dist_matrix_ = gower.gower_matrix(X_gower_)

    kmedoids_ = KMedoids(n_clusters=N_CLUSTERS, random_state=0, metric='precomputed', init='k-medoids++').fit(gower_dist_matrix_)
    distances_=[]
    gower_dist_=[]
    counts_=[]
    ideal_len_=len(kmedoids.labels_)/5

    for i in np.arange(N_CLUSTERS):
        cluster_data_ = X_gower_.loc[kmedoids_.labels_==i,:]
        # Compute the Gower distance between each data point in the cluster and each data point in the global dataset
        distances_matrix_ = gower.gower_matrix(cluster_data_, X_gower_)
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
    labels_.index=X_train.index
    close_index_train=labels_.index[np.where(labels_==False)[0]]
    far_index_train=labels_.index[np.where(labels_==True)[0]]


    # Convert data to PyTorch tensors
    # Modify X_train_, X_val, X_train, and X_test to have dummy variables
    X = pd.get_dummies(X.astype(str), drop_first=True)

    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    X_train_ = X_train.loc[close_index_train,:]
    X_val = X_train.loc[far_index_train,:]
    y_train_ = y_train.loc[close_index_train]
    y_val = y_train.loc[far_index_train]

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

    #### Gaussian process
    # Define the kernels
    kernels = [
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=X_train_.shape[1])),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=X_train_.shape[1])),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X_train_.shape[1])),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X_train_.shape[1])),
    ]

    best_crps = float('inf')
    best_kernel = None


    for kernel in kernels:
        # Initialize the Gaussian Process model and likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X_train__tensor, y_train__tensor, likelihood, kernel)

        if torch.cuda.is_available():
            model = model.cuda()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Train the model
        model.train()
        likelihood.train()

        train_GP(model,X_train__tensor,y_train__tensor,GP_ITERATIONS,mll,optimizer)
        
        # Set the model in evaluation mode
        model.eval()
        likelihood.eval()

        # Make predictions on the validation set
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_pred = model(X_val_tensor)

        # Calculate CRPS
        y_pred_np = y_pred.mean.cpu().numpy().flatten()
        y_pred_std_np = y_pred.stddev.cpu().numpy().flatten()

        # Calculate the CRPS for each prediction
        crps_values = [crps_gaussian(y_val_np[i], mu=y_pred_np[i], sig=y_pred_std_np[i]) for i in range(len(y_val_np))]

        # Calculate the mean CRPS
        mean_crps = np.mean(crps_values)

        # Update the best kernel if the current kernel has a lower CRPS
        if mean_crps < best_crps:
            best_crps = mean_crps
            best_kernel = kernel


    # Initialize the Gaussian Process model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train_tensor, y_train_tensor, likelihood, best_kernel)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if torch.cuda.is_available():
        model = model.cuda()

    # Train the model
    model.train()
    likelihood.train()
    train_GP(model,X_train_tensor,y_train_tensor,GP_ITERATIONS,mll,optimizer)

    # Set the model in evaluation mode
    model.eval()
    likelihood.eval()

    # Make predictions on the validation set
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_pred = model(X_test_tensor)

    # Calculate CRPS
    y_pred_np = y_pred.mean.cpu().numpy().flatten()
    y_pred_std_np = y_pred.stddev.cpu().numpy().flatten()

    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_pred_np[i], sig=y_pred_std_np[i]) for i in range(len(y_test_np))]

    # Calculate the mean CRPS
    CRPS_GP = np.mean(crps_values)

    # Update the best kernel if the current kernel has a lower CRPS
    print('CRPS_GP: ', CRPS_GP)


    # #### MLP
    d_out = 1  
    d_in=X_train_.shape[1]

    def MLP_opt(trial):

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        n_blocks = trial.suggest_int("n_blocks", 1, 5)
        d_block = trial.suggest_int("d_block", 10, 500)
        dropout = trial.suggest_float("dropout", 0, 1)

        MLP_model = MLP(
        d_in=d_in,
        d_out=d_out,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
        )
        n_epochs=N_EPOCHS
        learning_rate=trial.suggest_float('learning_rate', 0.0001, 0.05, log=True)
        weight_decay=trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
        optimizer=torch.optim.Adam(MLP_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()

        if torch.cuda.is_available():
            MLP_model = MLP_model.cuda()

        early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=CHECKPOINT_PATH)
        n_epochs=train(MLP_model, criterion, optimizer, n_epochs, train__loader, val_loader, early_stopping, CHECKPOINT_PATH)
        n_epochs = trial.suggest_int('n_epochs', n_epochs, n_epochs)

        # Point prediction
        predictions = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_predictions = MLP_model(batch_X).reshape(-1,)
                predictions.append(batch_predictions.cpu().numpy())

        y_val_hat_MLP = np.concatenate(predictions)

        # Estimate standard deviation of the prediction error
        std_dev_error = np.std(y_val - y_val_hat_MLP)

        # Calculate the CRPS for each prediction
        crps_values = [crps_gaussian(y_val_np[i], mu=y_val_hat_MLP[i], sig=std_dev_error) for i in range(len(y_val_hat_MLP))]

        # Calculate the mean CRPS
        mean_crps = np.mean(crps_values)

        return mean_crps

    sampler_MLP = optuna.samplers.TPESampler(seed=seed)
    study_MLP = optuna.create_study(sampler=sampler_MLP, direction='minimize')
    study_MLP.optimize(MLP_opt, n_trials=N_TRIALS)

    MLP_model = MLP(
        d_in=d_in,
        d_out=d_out,
        n_blocks=study_MLP.best_params['n_blocks'],
        d_block=study_MLP.best_params['d_block'],
        dropout=study_MLP.best_params['dropout'],
        )

    if torch.cuda.is_available():
        MLP_model = MLP_model.cuda()
        
    n_epochs=study_MLP.best_params['n_epochs']
    learning_rate=study_MLP.best_params['learning_rate']
    weight_decay=study_MLP.best_params['weight_decay']
    optimizer=torch.optim.Adam(MLP_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    train_no_early_stopping(MLP_model, criterion, optimizer, n_epochs, train_loader)

    # Point prediction
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_predictions = MLP_model(batch_X).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_test_hat_MLP = np.concatenate(predictions)

    # Estimate standard deviation of the prediction error
    std_dev_error = np.std(y_test - y_test_hat_MLP)

    # Create a normal distribution for each prediction
    pred_distributions = [norm(loc=y_test_hat_MLP[i], scale=std_dev_error) for i in range(len(y_test_hat_MLP))]

    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_test_hat_MLP[i], sig=std_dev_error) for i in range(len(y_test_hat_MLP))]

    # Calculate the mean CRPS
    crps_MLP = np.mean(crps_values)

    print("CRPS MLP: ", crps_MLP)

    # #### ResNet
    d_out = 1  
    d_in=X_train_.shape[1]

    def ResNet_opt(trial):

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        n_blocks = trial.suggest_int("n_blocks", 1, 5)
        d_block = trial.suggest_int("d_block", 10, 500)
        dropout1 = trial.suggest_float("dropout1", 0, 1)
        dropout2 = trial.suggest_float("dropout2", 0, 1)
        d_hidden_multiplier=trial.suggest_float("d_hidden_multiplier", 0.5, 3)

        ResNet_model = ResNet(
        d_in=d_in,
        d_out=d_out,
        n_blocks=n_blocks,
        d_block=d_block,
        d_hidden=None,
        d_hidden_multiplier=d_hidden_multiplier,
        dropout1=dropout1,
        dropout2=dropout2,
        )
        if torch.cuda.is_available():
            ResNet_model = ResNet_model.cuda()
        n_epochs=N_EPOCHS
        learning_rate=trial.suggest_float('learning_rate', 0.0001, 0.05, log=True)
        weight_decay=trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
        optimizer=torch.optim.Adam(ResNet_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()

        early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=CHECKPOINT_PATH)
        n_epochs=train(ResNet_model, criterion, optimizer, n_epochs, train__loader, val_loader, early_stopping, CHECKPOINT_PATH)
        n_epochs = trial.suggest_int('n_epochs', n_epochs, n_epochs)

        # Point prediction
        predictions = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_predictions = ResNet_model(batch_X).reshape(-1,)
                predictions.append(batch_predictions.cpu().numpy())

        y_val_hat_ResNet = np.concatenate(predictions)

        # Estimate standard deviation of the prediction error
        std_dev_error = np.std(y_val - y_val_hat_ResNet)

        # Calculate the CRPS for each prediction
        crps_values = [crps_gaussian(y_val_np[i], mu=y_val_hat_ResNet[i], sig=std_dev_error) for i in range(len(y_val_hat_ResNet))]

        # Calculate the mean CRPS
        crps_ResNet = np.mean(crps_values)

        return crps_ResNet

    sampler_ResNet = optuna.samplers.TPESampler(seed=seed)
    study_ResNet = optuna.create_study(sampler=sampler_ResNet, direction='minimize')
    study_ResNet.optimize(ResNet_opt, n_trials=N_TRIALS)

    ResNet_model = ResNet(
        d_in=d_in,
        d_out=d_out,
        n_blocks=study_ResNet.best_params['n_blocks'],
        d_block=study_ResNet.best_params['d_block'],
        d_hidden=None,
        d_hidden_multiplier=study_ResNet.best_params['d_hidden_multiplier'],
        dropout1=study_ResNet.best_params['dropout1'],
        dropout2=study_ResNet.best_params['dropout2'],
        )

    if torch.cuda.is_available():
        ResNet_model = ResNet_model.cuda()

    n_epochs=study_ResNet.best_params['n_epochs']
    learning_rate=study_ResNet.best_params['learning_rate']
    weight_decay=study_ResNet.best_params['weight_decay']
    optimizer=torch.optim.Adam(ResNet_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    train_no_early_stopping(ResNet_model, criterion, optimizer, n_epochs, train_loader)

    # Point prediction
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_predictions = ResNet_model(batch_X).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_test_hat_ResNet = np.concatenate(predictions)

    # Estimate standard deviation of the prediction error
    std_dev_error = np.std(y_test - y_test_hat_ResNet)

    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_test_hat_ResNet[i], sig=std_dev_error) for i in range(len(y_test_hat_ResNet))]

    # Calculate the mean CRPS
    crps_ResNet = np.mean(crps_values)

    print("CRPS ResNet: ", crps_ResNet)
    # #### FFTransformer

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

        if torch.cuda.is_available():
            FTTrans_model = FTTrans_model.cuda()

        n_epochs=N_EPOCHS
        learning_rate=trial.suggest_float('learning_rate', 0.0001, 0.05, log=True)
        weight_decay=trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
        optimizer=torch.optim.Adam(FTTrans_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()

        early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=CHECKPOINT_PATH)
        n_epochs=train_trans(FTTrans_model, criterion, optimizer, n_epochs, train__loader, val_loader, early_stopping, CHECKPOINT_PATH)
        n_epochs = trial.suggest_int('n_epochs', n_epochs, n_epochs)

        # Point prediction
        predictions = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_predictions = FTTrans_model(batch_X, None).reshape(-1,)
                predictions.append(batch_predictions.cpu().numpy())

        y_val_hat_FTTrans = np.concatenate(predictions)

        # Estimate standard deviation of the prediction error
        std_dev_error = np.std(y_val - y_val_hat_FTTrans)

        # Calculate the CRPS for each prediction
        crps_values = [crps_gaussian(y_val_np[i], mu=y_val_hat_FTTrans[i], sig=std_dev_error) for i in range(len(y_val_hat_FTTrans))]

        # Calculate the mean CRPS
        crps_FTTrans= np.mean(crps_values)

        return crps_FTTrans

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

    train_trans_no_early_stopping(FTTrans_model, criterion, optimizer, n_epochs, train_loader)

    # Point prediction
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_predictions = FTTrans_model(batch_X, None).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_test_hat_FTTrans = np.concatenate(predictions)

    # Estimate standard deviation of the prediction error
    std_dev_error = np.std(y_test - y_test_hat_FTTrans)

    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_test_hat_FTTrans[i], sig=std_dev_error) for i in range(len(y_test_hat_FTTrans))]

    # Calculate the mean CRPS
    crps_FTTrans= np.mean(crps_values)

    print("CRPS FTTrans: ", crps_FTTrans)

    # #### Boosted trees, random forest, engression, linear regression
    # Create lgb dataset
    dtrain_ = lgb.Dataset(torch.tensor(X_train_.values, dtype=torch.float32).clone().detach(), label=y_train_.values)

    def boosted(trial):

        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.5, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        }
        opt_params = params.copy()
        n_rounds = opt_params["n_estimators"]
        del opt_params["n_estimators"]
        opt_params['feature_pre_filter']=False

        # Use LightGBMLossGuideRegressor for distributional prediction
        boosted_tree_model = LightGBMLSS(Gaussian(stabilization="None", response_fn="exp", loss_fn="nll"))
        boosted_tree_model.train(opt_params, dtrain_, num_boost_round=n_rounds)

        # Predict both the mean and standard deviation
        pred_params=boosted_tree_model.predict(X_val, pred_type="parameters")
        y_val_hat_boost=pred_params['loc']
        y_val_hat_std = pred_params['scale']

        # Calculate the CRPS for each prediction
        crps_values = [crps_gaussian(y_val_np[i], mu=y_val_hat_boost[i], sig=y_val_hat_std[i]) for i in range(len(y_val))]

        # Return the mean CRPS as the objective to be minimized
        return np.mean(crps_values)

    sampler_boost = optuna.samplers.TPESampler(seed=seed)
    study_boost = optuna.create_study(sampler=sampler_boost, direction='minimize')
    study_boost.optimize(boosted, n_trials=N_TRIALS)

    np.random.seed(seed)
    quantiles=list(np.random.uniform(0,1,N_SAMPLES))
    def rf(trial):
        params = {'num_trees': trial.suggest_int('num_trees', 100, 500),
            'mtry': trial.suggest_int('mtry', 1, 30),
            'min_node_size': trial.suggest_int('min_node_size', 10, 100)}
        
        drf_model = drf(**params, seed=seed)
        drf_model.fit(X_train_, y_train_)
        
        # Generate a sample from the drf model for each data point
        y_val_hat=drf_model.predict(newdata = X_val, functional = "quantile", quantiles=quantiles)

        # Calculate the CRPS for each prediction
        crps_values = [crps_ensemble(y_val_np[i], y_val_hat.quantile[i].reshape(-1)) for i in range(len(y_val_np))]

        # Return the mean CRPS as the objective to be minimized
        return np.mean(crps_values)

    sampler_drf = optuna.samplers.TPESampler(seed=seed)
    study_drf = optuna.create_study(sampler=sampler_drf, direction='minimize')
    study_drf.optimize(rf, n_trials=N_TRIALS)


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
        crps_values = [crps_ensemble(y_val_np[i], np.array(y_val_hat_engression_samples[i]).reshape(-1,)) for i in range(len(y_val_np))]

        return np.mean(crps_values)

    sampler_engression = optuna.samplers.TPESampler(seed=seed)
    study_engression = optuna.create_study(sampler=sampler_engression, direction='minimize')
    study_engression.optimize(engressor_NN, n_trials=N_TRIALS)


    dtrain = lgb.Dataset(torch.tensor(X_train.values, dtype=torch.float32).clone().detach(), label=y_train.values)
    opt_params = study_boost.best_params.copy()
    n_rounds = opt_params["n_estimators"]
    del opt_params["n_estimators"]
    opt_params['feature_pre_filter']=False
    # Use LightGBMLossGuideRegressor for distributional prediction
    boosted_tree_model = LightGBMLSS(Gaussian(stabilization="None", response_fn="exp", loss_fn="nll"))
    boosted_tree_model.train(opt_params, dtrain, num_boost_round=n_rounds)
    # Predict both the mean and standard deviation
    pred_params=boosted_tree_model.predict(X_test, pred_type="parameters")
    y_test_hat_boost=pred_params['loc']
    y_test_hat_std = pred_params['scale']
    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_test_hat_boost[i], sig=y_test_hat_std[i]) for i in range(len(y_test))]
    # Return the mean CRPS as the objective to be minimized
    CRPS_boosted=np.mean(crps_values)

    drf_model=drf(**study_drf.best_params, seed=seed)
    drf_model.fit(X_train, y_train)
    # Generate a sample from the drf model for each data point
    y_test_hat_drf=drf_model.predict(newdata = X_test, functional = "quantile", quantiles=quantiles)
    # Calculate the CRPS for each prediction
    crps_values = [crps_ensemble(y_test_np[i], y_test_hat_drf.quantile[i].reshape(-1)) for i in range(len(y_test_np))]
    # Return the mean CRPS as the objective to be minimized
    CRPS_rf=np.mean(crps_values)

    lin_reg=LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_test_hat_linreg=lin_reg.predict(X_test)
    # Calculate the standard deviation of the residuals
    std_dev = np.std(y_test - y_test_hat_linreg)
    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_test_hat_linreg[i], sig=std_dev) for i in range(len(y_test_np))]
    CRPS_linreg = np.mean(crps_values)

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
    crps_values = [crps_ensemble(y_test_np[i], np.array(y_test_hat_engression_samples[i]).reshape(-1,)) for i in range(len(y_test_np))]
    CRPS_engression=np.mean(crps_values)

    print("CRPS linear regression: ",CRPS_linreg)
    print("CRPS boosted trees", CRPS_boosted)
    print("CRPS random forest", CRPS_rf)
    print("CRPS engression", CRPS_engression)

    #### GAM model
    def gam_model(trial):

        # Define the hyperparameters to optimize
        params = {'n_splines': trial.suggest_int('n_splines', 5, 20),
                'lam': trial.suggest_loguniform('lam', 1e-3, 1)}

        # Create and train the model
        gam = LinearGAM(s(0, n_splines=params['n_splines'], lam=params['lam'])).fit(X_train_, y_train_)

        # Predict on the validation set and calculate the CRPS
        y_val_hat_gam = gam.predict(X_val)
        std_dev_error = np.std(y_val - y_val_hat_gam)
        crps_gam = [crps_gaussian(y_val_np[i], mu=y_val_hat_gam[i], sig=std_dev_error) for i in range(len(y_val_hat_gam))]
        crps_gam = np.mean(crps_gam)

        return crps_gam

    # Create the sampler and study
    sampler_gam = optuna.samplers.TPESampler(seed=seed)
    study_gam = optuna.create_study(sampler=sampler_gam, direction='minimize')

    # Optimize the model
    study_gam.optimize(gam_model, n_trials=N_TRIALS)

    # Create the final model with the best parameters
    best_params = study_gam.best_params
    final_gam_model = LinearGAM(s(0, n_splines=best_params['n_splines'], lam=best_params['lam']))

    # Fit the model
    final_gam_model.fit(X_train, y_train)

    # Predict on the test set
    y_test_hat_gam = final_gam_model.predict(X_test)

    # Calculate the CRPS
    std_dev_error = np.std(y_test - y_test_hat_gam)
    crps_gam = [crps_gaussian(y_test_np[i], mu=y_test_hat_gam[i], sig=std_dev_error) for i in range(len(y_test_hat_gam))]
    crps_gam = np.mean(crps_gam)
    print("CRPS GAM: ", crps_gam)

    crps_results = {'GP': CRPS_GP, 'MLP': crps_MLP, 'ResNet': crps_ResNet, 'FTTrans': crps_FTTrans, 'boosted_trees': CRPS_boosted, 'drf': CRPS_rf, 'linear_regression': CRPS_linreg, 'engression': CRPS_engression, 'GAM': crps_gam}  # Add all your methods here

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(crps_results.items()), columns=['Method', 'CRPS'])

    # Create the directory if it doesn't exist
    os.makedirs('RESULTS/K_MEDOIDS', exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f'RESULTS/K_MEDOIDS/{task_id}_k_medoids_crps_results.csv', index=False)
