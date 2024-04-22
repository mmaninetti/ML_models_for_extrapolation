from umap import UMAP
import os
import pandas as pd
import numpy as np
import openml
from sklearn.linear_model import LinearRegression 
import lightgbm as lgbm
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern
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
from sklearn.metrics.pairwise import euclidean_distances
from utils import EarlyStopping, train, train_trans, train_no_early_stopping, train_trans_no_early_stopping
from torch.utils.data import TensorDataset, DataLoader
import shutil
import gpboost as gpb

# Create the checkpoint directory if it doesn't exist
if os.path.exists('CHECKPOINTS/UMAP'):
    shutil.rmtree('CHECKPOINTS/UMAP')
os.makedirs('CHECKPOINTS/UMAP')

#openml.config.apikey = 'FILL_IN_OPENML_API_KEY'  # set the OpenML Api Key
#SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

#task_id=361093
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

    CHECKPOINT_PATH = f'CHECKPOINTS/UMAP/task_{task_id}.pt'

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


    # Apply UMAP decomposition
    umap = UMAP(n_components=2, random_state=42)
    X_umap = umap.fit_transform(X_clean)

    # calculate the Euclidean distance matrix
    euclidean_dist_matrix = euclidean_distances(X_umap)

    # calculate the Euclidean distance for each data point
    euclidean_dist = np.mean(euclidean_dist_matrix, axis=1)

    euclidean_dist = pd.Series(euclidean_dist, index=X_clean.index)
    far_index = euclidean_dist.index[np.where(euclidean_dist >= np.quantile(euclidean_dist, 0.8))[0]]
    close_index = euclidean_dist.index[np.where(euclidean_dist < np.quantile(euclidean_dist, 0.8))[0]]

    X_train_clean = X_clean.loc[close_index,:]
    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    # Apply UMAP decomposition on the training set
    X_umap_train = umap.fit_transform(X_train_clean)

    # calculate the Euclidean distance matrix for the training set
    euclidean_dist_matrix_train = euclidean_distances(X_umap_train)

    # calculate the Euclidean distance for each data point in the training set
    euclidean_dist_train = np.mean(euclidean_dist_matrix_train, axis=1)

    euclidean_dist_train = pd.Series(euclidean_dist_train, index=X_train_clean.index)
    far_index_train = euclidean_dist_train.index[np.where(euclidean_dist_train >= np.quantile(euclidean_dist_train, 0.8))[0]]
    close_index_train = euclidean_dist_train.index[np.where(euclidean_dist_train < np.quantile(euclidean_dist_train, 0.8))[0]]


    # Convert data to PyTorch tensors
    # Modify X_train_, X_val, X_train, and X_test to have dummy variables
    X = pd.get_dummies(X, drop_first=True)

    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    X_train_ = X_train.loc[close_index_train,:]
    X_val = X_train.loc[far_index_train,:]
    y_train_ = y_train.loc[close_index_train]
    y_val = y_train.loc[far_index_train]

    # Standardize the data for non-dummy variables
    non_dummy_cols = X.select_dtypes(exclude=['bool']).columns
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

    # #### MLP
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
            for batch_X, _ in train__loader:
                batch_predictions = MLP_model(batch_X).reshape(-1,)
                predictions.append(batch_predictions.cpu().numpy())

        y_train__hat_MLP = np.concatenate(predictions)

        # Estimate standard deviation of the prediction error
        std_dev_error = np.std(y_train_ - y_train__hat_MLP)

        # Point prediction
        predictions = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_predictions = MLP_model(batch_X).reshape(-1,)
                predictions.append(batch_predictions.cpu().numpy())

        y_val_hat_MLP = np.concatenate(predictions)

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
        for batch_X, _ in train_loader:
            batch_predictions = MLP_model(batch_X).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_train_hat_MLP = np.concatenate(predictions)

    # Estimate standard deviation of the prediction error
    std_dev_error = np.std(y_train - y_train_hat_MLP)

    # Point prediction
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_predictions = MLP_model(batch_X).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_test_hat_MLP = np.concatenate(predictions)

    # Create a normal distribution for each prediction
    pred_distributions = [norm(loc=y_test_hat_MLP[i], scale=std_dev_error) for i in range(len(y_test_hat_MLP))]

    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_test_hat_MLP[i], sig=std_dev_error) for i in range(len(y_test_hat_MLP))]

    # Calculate the mean CRPS
    crps_MLP = np.mean(crps_values)

    print("CRPS MLP: ", crps_MLP)
    del MLP_model, optimizer, criterion, y_test_hat_MLP, predictions

    # #### ResNet
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
            for batch_X, _ in train__loader:
                batch_predictions = ResNet_model(batch_X).reshape(-1,)
                predictions.append(batch_predictions.cpu().numpy())

        y_train__hat_ResNet = np.concatenate(predictions)

        # Estimate standard deviation of the prediction error
        std_dev_error = np.std(y_train_ - y_train__hat_ResNet)

        # Point prediction
        predictions = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_predictions = ResNet_model(batch_X).reshape(-1,)
                predictions.append(batch_predictions.cpu().numpy())

        y_val_hat_ResNet = np.concatenate(predictions)

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
        for batch_X, _ in train_loader:
            batch_predictions = ResNet_model(batch_X).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_train_hat_ResNet = np.concatenate(predictions)

    # Estimate standard deviation of the prediction error
    std_dev_error = np.std(y_train - y_train_hat_ResNet)

    # Point prediction
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_predictions = ResNet_model(batch_X).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_test_hat_ResNet = np.concatenate(predictions)

    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_test_hat_ResNet[i], sig=std_dev_error) for i in range(len(y_test_hat_ResNet))]

    # Calculate the mean CRPS
    crps_ResNet = np.mean(crps_values)

    print("CRPS ResNet: ", crps_ResNet)
    del ResNet_model, optimizer, criterion, y_test_hat_ResNet, predictions
    
    #### FFTransformer
    #### Skip for now, will do it later
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
            for batch_X, _ in train__loader:
                batch_predictions = FTTrans_model(batch_X, None).reshape(-1,)
                predictions.append(batch_predictions.cpu().numpy())

        y_train__hat_FTTrans = np.concatenate(predictions)

        # Estimate standard deviation of the prediction error
        std_dev_error = np.std(y_train_ - y_train__hat_FTTrans)
        
        # Point prediction
        predictions = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_predictions = FTTrans_model(batch_X, None).reshape(-1,)
                predictions.append(batch_predictions.cpu().numpy())

        y_val_hat_FTTrans = np.concatenate(predictions)

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
        for batch_X, _ in train_loader:
            batch_predictions = FTTrans_model(batch_X, None).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_train_hat_FTTrans = np.concatenate(predictions)

    # Estimate standard deviation of the prediction error
    std_dev_error = np.std(y_train - y_train_hat_FTTrans)
    
    # Point prediction
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_predictions = FTTrans_model(batch_X, None).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_test_hat_FTTrans = np.concatenate(predictions)

    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_test_hat_FTTrans[i], sig=std_dev_error) for i in range(len(y_test_hat_FTTrans))]

    # Calculate the mean CRPS
    crps_FTTrans= np.mean(crps_values)

    print("CRPS FTTrans: ", crps_FTTrans)
    del FTTrans_model, optimizer, criterion, y_test_hat_FTTrans, predictions
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    else:
        print("The file does not exist.")

    #### Boosted trees, random forest, engression, linear regression
    # Create lgb dataset
    dtrain_ = lgb.Dataset(torch.tensor(X_train_.values, dtype=torch.float32).clone().detach(), label=y_train_.values)

    def boosted_LSS(trial):

        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.5, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'num_leaves': 2**10,
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        }
        opt_params = params.copy()
        n_rounds = opt_params["n_estimators"]
        del opt_params["n_estimators"]
        opt_params['feature_pre_filter']=False

        # Use LightGBMLossGuideRegressor for distributional prediction
        boosted_tree_model = LightGBMLSS(Gaussian(stabilization="None", response_fn="exp", loss_fn="crps"))
        boosted_tree_model.train(opt_params, dtrain_, num_boost_round=n_rounds)

        # Predict both the mean and standard deviation
        pred_params=boosted_tree_model.predict(X_val, pred_type="parameters")
        y_val_hat_boost=pred_params['loc']
        y_val_hat_std = pred_params['scale']

        # Calculate the CRPS for each prediction
        crps_values = [crps_gaussian(y_val_np[i], mu=y_val_hat_boost[i], sig=y_val_hat_std[i]) for i in range(len(y_val))]

        # Return the mean CRPS as the objective to be minimized
        return np.mean(crps_values)

    sampler_boost_LSS = optuna.samplers.TPESampler(seed=seed)
    study_boost_LSS = optuna.create_study(sampler=sampler_boost_LSS, direction='minimize')
    study_boost_LSS.optimize(boosted_LSS, n_trials=N_TRIALS)

    np.random.seed(seed)
    quantiles=list(np.random.uniform(0,1,N_SAMPLES))
    def distributional_rf(trial):
        params = {'num_trees': trial.suggest_int('num_trees', 100, 500),
            'mtry': trial.suggest_int('mtry', 0, d_in),
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
    study_drf.optimize(distributional_rf, n_trials=N_TRIALS)

    def boosted(trial):

        params = {'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.5, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 1, 30),
                'num_leaves': 2**10,
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100)}
        
        boosted_tree_model=lgbm.LGBMRegressor(**params)
        boosted_tree_model.fit(X_train_, y_train_)
        y_train__hat_boost=boosted_tree_model.predict(X_train_)
        
        std_dev = np.std(y_train_ - y_train__hat_boost)
        y_val_hat_boost=boosted_tree_model.predict(X_val)
        # Calculate the CRPS for each prediction
        crps_values = [crps_gaussian(y_val_np[i], mu=y_val_hat_boost[i], sig=std_dev) for i in range(len(y_val_np))]
        CRPS_boost = np.mean(crps_values)

        return CRPS_boost

    sampler_boost = optuna.samplers.TPESampler(seed=seed)
    study_boost = optuna.create_study(sampler=sampler_boost, direction='minimize')
    study_boost.optimize(boosted, n_trials=N_TRIALS)
    params=study_boost.best_params
    params['num_leaves']=2**10
    boosted_model=lgbm.LGBMRegressor(**params)

    def rf(trial):

        params = {'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 1, 30),
                'max_features': trial.suggest_float('max_features', 0, 1),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100)}
        
        rf_model=RandomForestRegressor(**params)
        rf_model.fit(X_train_, y_train_)
        
        y_train__hat_rf=rf_model.predict(X_train_)
        std_dev = np.std(y_train_ - y_train__hat_rf)
        y_val_hat_rf=rf_model.predict(X_val)

        # Calculate the CRPS for each prediction
        crps_values = [crps_gaussian(y_val_np[i], mu=y_val_hat_rf[i], sig=std_dev) for i in range(len(y_val_np))]
        CRPS_rf = np.mean(crps_values)

        return CRPS_rf

    sampler_rf = optuna.samplers.TPESampler(seed=seed)
    study_rf = optuna.create_study(sampler=sampler_rf, direction='minimize')
    study_rf.optimize(rf, n_trials=N_TRIALS)
    rf_model=RandomForestRegressor(**study_rf.best_params)


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

        return np.mean(crps_values)

    sampler_engression = optuna.samplers.TPESampler(seed=seed)
    study_engression = optuna.create_study(sampler=sampler_engression, direction='minimize')
    study_engression.optimize(engressor_NN, n_trials=N_TRIALS)


    dtrain = lgb.Dataset(torch.tensor(X_train.values, dtype=torch.float32).clone().detach(), label=y_train.values)
    opt_params = study_boost_LSS.best_params.copy()
    opt_params['num_leaves']=2**10
    n_rounds = opt_params["n_estimators"]
    del opt_params["n_estimators"]
    opt_params['feature_pre_filter']=False
    # Use LightGBMLossGuideRegressor for distributional prediction
    boosted_tree_model_LSS = LightGBMLSS(Gaussian(stabilization="None", response_fn="exp", loss_fn="crps"))
    boosted_tree_model_LSS.train(opt_params, dtrain, num_boost_round=n_rounds)
    # Predict both the mean and standard deviation
    pred_params=boosted_tree_model_LSS.predict(X_test, pred_type="parameters")
    y_test_hat_boost_LSS = pred_params['loc']
    y_test_hat_std = pred_params['scale']
    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_test_hat_boost_LSS[i], sig=y_test_hat_std[i]) for i in range(len(y_test))]
    # Return the mean CRPS as the objective to be minimized
    CRPS_boosted_LSS=np.mean(crps_values)

    boosted_model.fit(X_train, y_train)
    y_train_hat_boosted=boosted_model.predict(X_train)
    # Calculate the standard deviation of the residuals
    std_dev = np.std(y_train - y_train_hat_boosted)
    y_test_hat_boosted=boosted_model.predict(X_test)
    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_test_hat_boosted[i], sig=std_dev) for i in range(len(y_test_np))]
    CRPS_boosted = np.mean(crps_values)
    

    rf_model.fit(X_train, y_train)
    y_train_hat_rf=rf_model.predict(X_train)
    # Calculate the standard deviation of the residuals
    std_dev = np.std(y_train - y_train_hat_rf)
    y_test_hat_rf=rf_model.predict(X_test)
    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=y_test_hat_rf[i], sig=std_dev) for i in range(len(y_test_np))]
    CRPS_rf = np.mean(crps_values)
    

    drf_model=drf(**study_drf.best_params, seed=seed)
    drf_model.fit(X_train, y_train)
    # Generate a sample from the drf model for each data point
    y_test_hat_drf=drf_model.predict(newdata = X_test, functional = "quantile", quantiles=quantiles)
    # Calculate the CRPS for each prediction
    crps_values = [crps_ensemble(y_test_np[i], y_test_hat_drf.quantile[i].reshape(-1)) for i in range(len(y_test_np))]
    # Return the mean CRPS as the objective to be minimized
    CRPS_drf=np.mean(crps_values)

    lin_reg=LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_train_hat_linreg=lin_reg.predict(X_train)
    # Calculate the standard deviation of the residuals
    std_dev = np.std(y_train - y_train_hat_linreg)
    y_test_hat_linreg=lin_reg.predict(X_test)
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
    crps_values = [crps_ensemble(y_test_np[i], np.array(y_test_hat_engression_samples[i].cpu()).reshape(-1,)) for i in range(len(y_test_np))]
    CRPS_engression=np.mean(crps_values)

    constant_prediction_train = np.array([np.mean(y_train)]*len(y_train))
    # Calculate the standard deviation of the residuals
    std_dev = np.std(y_train - constant_prediction_train)
    constant_prediction = np.array([np.mean(y_train)]*len(y_test))
    # Calculate the CRPS for each prediction
    crps_values = [crps_gaussian(y_test_np[i], mu=constant_prediction[i], sig=std_dev) for i in range(len(y_test_np))]
    CRPS_constant = np.mean(crps_values)

    print("CRPS linear regression: ",CRPS_linreg)
    print("CRPS distributional boosted trees", CRPS_boosted_LSS)
    print("CRPS distributional random forest", CRPS_drf)
    print("CRPS boosted trees", CRPS_boosted)
    print("CRPS random forest", CRPS_rf)
    print("CRPS engression", CRPS_engression)
    print("CRPS constant", CRPS_constant)

    #### GAM model
    # Calculate the crps
    # I will add it later
    crps_gam = float("NaN")
    print("CRPS GAM: ", crps_gam)
    print("CRPS GAM: ", crps_gam)

    #### GP model
    # For now, just set it to Nan
    CRPS_GP = float("NaN")

    crps_results = {'constant': CRPS_constant, 'MLP': crps_MLP, 'ResNet': crps_ResNet, 'FTTrans': crps_FTTrans, 'distributional_boosted_trees': CRPS_boosted_LSS, 'drf': CRPS_drf, 'boosted_trees': CRPS_boosted, 'rf': CRPS_rf, 'linear_regression': CRPS_linreg, 'engression': CRPS_engression, 'GAM': crps_gam, 'GP': CRPS_GP}
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(crps_results.items()), columns=['Method', 'CRPS'])

    # Create the directory if it doesn't exist
    os.makedirs('RESULTS/UMAP_DECOMPOSITION', exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f'RESULTS/UMAP_DECOMPOSITION/{task_id}_umap_decomposition_crps_results.csv', index=False)
