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
from sklearn_extra.cluster import KMedoids
import gower
from sklearn.preprocessing import LabelEncoder
from utils import EarlyStopping, train, train_trans, train_no_early_stopping, train_trans_no_early_stopping
from torch.utils.data import TensorDataset, DataLoader
import re
import shutil
import gpboost as gpb

# Create the checkpoint directory if it doesn't exist
if os.path.exists('CHECKPOINTS/KMEDOIDS'):
    shutil.rmtree('CHECKPOINTS/KMEDOIDS')
os.makedirs('CHECKPOINTS/KMEDOIDS')

#SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

#task_id=361110
for task_id in benchmark_suite.tasks[1:]:  # iterate over all tasks in the suite

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
    X_val_tensor = torch.tensor(X_val_tensor.values, dtype=torch.float32)
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

    #### MLP
    def MLP_opt(trial):

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        n_blocks = trial.suggest_int("n_blocks", 1, 5)
        d_block = trial.suggest_int("d_block", 10, 500)
        dropout = trial.suggest_float("dropout", 0, 1)

        MLP_model = MLP(
        d_in=d_in,
        d_out=1,  # For binary classification, output dimension should be 1
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
        )
        n_epochs=N_EPOCHS
        learning_rate=trial.suggest_float('learning_rate', 0.0001, 0.05, log=True)
        weight_decay=trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
        optimizer=torch.optim.Adam(MLP_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()  # Use Binary Cross Entropy loss for binary classification

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
        y_val_hat_MLP = torch.sigmoid(torch.tensor(np.concatenate(predictions)))  # Apply sigmoid to get probabilities
        accuracy_MLP = accuracy_score(y_val_tensor.cpu().numpy(), y_val_hat_MLP.ge(0.5).float().cpu().numpy())  # Calculate accuracy

        return accuracy_MLP

    sampler_MLP = optuna.samplers.TPESampler(seed=seed)
    study_MLP = optuna.create_study(sampler=sampler_MLP, direction='maximize')  # We want to maximize accuracy
    study_MLP.optimize(MLP_opt, n_trials=N_TRIALS)

    MLP_model = MLP(
        d_in=d_in,
        d_out=1,  # For binary classification, output dimension should be 1
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
    criterion = torch.nn.BCEWithLogitsLoss()  # Use Binary Cross Entropy loss for binary classification

    train_no_early_stopping(MLP_model, criterion, optimizer, n_epochs, train_loader)

    # Point prediction
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_predictions = MLP_model(batch_X).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_test_hat_MLP = torch.sigmoid(torch.Tensor(np.concatenate(predictions)))
    accuracy_MLP = accuracy_score(y_test_tensor.cpu().numpy(), y_test_hat_MLP.ge(0.5).float().cpu().numpy())  # Calculate accuracy
    print("Accuracy MLP: ", accuracy_MLP)
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
        d_out=1,  # For binary classification, output dimension should be 1
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
        criterion = torch.nn.BCEWithLogitsLoss()  # Use Binary Cross Entropy loss for binary classification

        early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=CHECKPOINT_PATH)
        n_epochs=train(ResNet_model, criterion, optimizer, n_epochs, train__loader, val_loader, early_stopping, CHECKPOINT_PATH)
        n_epochs = trial.suggest_int('n_epochs', n_epochs, n_epochs)

        # Point prediction
        predictions = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_predictions = ResNet_model(batch_X).reshape(-1,)
                predictions.append(batch_predictions.cpu().numpy())

        y_val_hat_ResNet = torch.sigmoid(torch.Tensor(np.concatenate(predictions)))  # Apply sigmoid to get probabilities
        accuracy_ResNet = accuracy_score(y_val_tensor.cpu().numpy(), y_val_hat_ResNet.ge(0.5).float().cpu().numpy())  # Calculate accuracy

        return accuracy_ResNet

    sampler_ResNet = optuna.samplers.TPESampler(seed=seed)
    study_ResNet = optuna.create_study(sampler=sampler_ResNet, direction='maximize')  # We want to maximize accuracy
    study_ResNet.optimize(ResNet_opt, n_trials=N_TRIALS)

    ResNet_model = ResNet(
        d_in=d_in,
        d_out=1,  # For binary classification, output dimension should be 1
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
    criterion = torch.nn.BCEWithLogitsLoss()  # Use Binary Cross Entropy loss for binary classification

    train_no_early_stopping(ResNet_model, criterion, optimizer, n_epochs, train_loader)

    # Point prediction
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_predictions = ResNet_model(batch_X).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_test_hat_ResNet = torch.sigmoid(torch.Tensor(np.concatenate(predictions)))  # Apply sigmoid to get probabilities
    accuracy_ResNet = accuracy_score(y_test_tensor.cpu().numpy(), y_test_hat_ResNet.ge(0.5).float().cpu().numpy())  # Calculate accuracy
    print("Accuracy ResNet: ", accuracy_ResNet)
    del ResNet_model, optimizer, criterion, y_test_hat_ResNet, predictions

    #### FFTransformer
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
        d_out=1,  # For binary classification, output dimension should be 1
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
        criterion = torch.nn.BCEWithLogitsLoss()  # Use Binary Cross Entropy loss for binary classification

        early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=CHECKPOINT_PATH)
        n_epochs=train_trans(FTTrans_model, criterion, optimizer, n_epochs, train__loader, val_loader, early_stopping, CHECKPOINT_PATH)
        n_epochs = trial.suggest_int('n_epochs', n_epochs, n_epochs)

        # Point prediction
        predictions = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_predictions = FTTrans_model(batch_X, None).reshape(-1,)
                predictions.append(batch_predictions.cpu().numpy())

        y_val_hat_FTTrans = torch.sigmoid(torch.Tensor(np.concatenate(predictions)))  # Apply sigmoid to get probabilities
        accuracy_FTTrans = accuracy_score(y_val_tensor.cpu().numpy(), y_val_hat_FTTrans.ge(0.5).float().cpu().numpy())  # Calculate accuracy

        return accuracy_FTTrans

    sampler_FTTrans = optuna.samplers.TPESampler(seed=seed)
    study_FTTrans = optuna.create_study(sampler=sampler_FTTrans, direction='maximize')  # We want to maximize accuracy
    study_FTTrans.optimize(FTTrans_opt, n_trials=N_TRIALS)


    FTTrans_model = FTTransformer(
        n_cont_features=d_in,
        cat_cardinalities=[],
        d_out=1,  # For binary classification, output dimension should be 1
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
    criterion = torch.nn.BCEWithLogitsLoss()  # Use Binary Cross Entropy loss for binary classification
    loss_Adam=[]

    train_trans_no_early_stopping(FTTrans_model, criterion, optimizer, n_epochs, train_loader)

    # Point prediction
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_predictions = FTTrans_model(batch_X, None).reshape(-1,)
            predictions.append(batch_predictions.cpu().numpy())

    y_test_hat_FTTrans = torch.sigmoid(torch.Tensor(np.concatenate(predictions)))  # Apply sigmoid to get probabilities
    accuracy_FTTrans = accuracy_score(y_test_tensor.cpu().numpy(), y_test_hat_FTTrans.ge(0.5).float().cpu().numpy())  # Calculate accuracy
    print("Accuracy FTTrans: ", accuracy_FTTrans)
    del FTTrans_model, optimizer, criterion, y_test_hat_FTTrans, predictions
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    else:
        print("The file does not exist.")

    #### Boosted trees, random forest, engression, linear regression
    def boosted(trial):

        params = {'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.5, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 1, 30),
                'num_leaves': 2**10,
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100)}
        
        boosted_tree_model=lgbm.LGBMClassifier(**params)
        boosted_tree_model.fit(X_train_, y_train_)
        y_val_hat_boost=boosted_tree_model.predict(X_val)
        print(y_val_hat_boost)
        accuracy_boost=accuracy_score(y_val, y_val_hat_boost)

        return accuracy_boost

    sampler_boost = optuna.samplers.TPESampler(seed=seed)
    study_boost = optuna.create_study(sampler=sampler_boost, direction='maximize')
    study_boost.optimize(boosted, n_trials=N_TRIALS)
    params=study_boost.best_params
    params['num_leaves']=2**10
    boosted_model=lgbm.LGBMClassifier(**params)

    def rf(trial):

        params = {'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 1, 30),
                'max_features': trial.suggest_float('max_features', 0, 1),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100)}
        
        rf_model=RandomForestClassifier(**params)
        rf_model.fit(X_train_, y_train_)
        y_val_hat_rf=rf_model.predict(X_val)
        accuracy_rf=accuracy_score(y_val, y_val_hat_rf)

        return accuracy_rf

    sampler_rf = optuna.samplers.TPESampler(seed=seed)
    study_rf = optuna.create_study(sampler=sampler_rf, direction='maximize')
    study_rf.optimize(rf, n_trials=N_TRIALS)
    rf_model=RandomForestClassifier(**study_rf.best_params)

    def engressor_NN(trial):

        params = {'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'num_epoches': trial.suggest_int('num_epoches', 100, 1000),
                'num_layer': trial.suggest_int('num_layer', 2, 5),
                'hidden_dim': trial.suggest_int('hidden_dim', 100, 500),
                'resblock': trial.suggest_categorical('resblock', [True, False])}
        params['noise_dim']=params['hidden_dim']

        # Check if CUDA is available and if so, move the tensors and the model to the GPU
        if torch.cuda.is_available():
            engressor_model=engression(X_train__tensor, y_train__tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=BATCH_SIZE, sigmoid=True, resblock=params['resblock'], device="cuda")
        else: 
            engressor_model=engression(X_train__tensor, y_train__tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=BATCH_SIZE, sigmoid=True, resblock=params['resblock'])
        
        # Generate a sample from the engression model for each data point
        y_val_hat_engression=engressor_model.predict(X_val_tensor, target="mean")
        y_val_hat_engression = y_val_hat_engression.ge(0.5).float()  # Apply threshold to get binary predictions

        accuracy_engression = accuracy_score(y_val_tensor.cpu().numpy(), y_val_hat_engression.cpu().numpy())  # Calculate accuracy

        return accuracy_engression

    sampler_engression = optuna.samplers.TPESampler(seed=seed)
    study_engression = optuna.create_study(sampler=sampler_engression, direction='maximize')  # We want to maximize accuracy
    study_engression.optimize(engressor_NN, n_trials=N_TRIALS)


    # Fit the boosted model and make predictions
    boosted_model.fit(X_train, y_train)
    y_test_hat_boosted = boosted_model.predict(X_test)
    accuracy_boosted = accuracy_score(y_test, y_test_hat_boosted)

    # Fit the random forest model and make predictions
    rf_model.fit(X_train, y_train)
    y_test_hat_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_test_hat_rf)

    # Fit the logistic regression model and make predictions
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_test_hat_logreg = log_reg.predict(X_test)
    accuracy_logreg = accuracy_score(y_test, y_test_hat_logreg)

    # Engression model
    params=study_engression.best_params
    params['noise_dim']=params['hidden_dim']
    # Check if CUDA is available and if so, move the tensors and the model to the GPU
    if torch.cuda.is_available():
        engressor_model=engression(X_train_tensor, y_train_tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=BATCH_SIZE, sigmoid=True, resblock=params['resblock'], device="cuda")
    else: 
        engressor_model=engression(X_train_tensor, y_train_tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=BATCH_SIZE, sigmoid=True, resblock=params['resblock'])
    # Assuming the model outputs probabilities for the two classes
    y_test_hat_engression=engressor_model.predict(X_test_tensor, target="mean")
    # Convert the probabilities to class labels
    y_test_hat_engression = y_test_hat_engression.ge(0.5).float()  # Apply threshold to get binary predictions
    accuracy_engression = accuracy_score(y_test_tensor.cpu().numpy(), y_test_hat_engression.cpu().numpy())  # Calculate accuracy

    constant_prediction = np.array([np.mean(y_train)]*len(y_test))
    constant_prediction = np.where(constant_prediction >= 0.5, 1, 0)
    accuracy_constant = accuracy_score(y_test, constant_prediction)

    print("Accuracy logistic regression: ", accuracy_logreg)
    print("Accuracy boosted trees: ", accuracy_boosted)
    print("Accuracy random forest: ", accuracy_rf)
    print("Accuracy engression: ", accuracy_engression)
    print("Accuracy constant prediction: ", accuracy_constant)

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

    #### GAM model
    # Calculate the accuracy
    # I will add it later
    accuracy_gam = float("NaN")
    print("Accuracy GAM: ", accuracy_gam)

    accuracy_results = {'Constant': accuracy_constant, 'MLP': accuracy_MLP, 'ResNet': accuracy_ResNet, 'FTTrans': accuracy_FTTrans, 'boosted_trees': accuracy_boosted, 'rf': accuracy_rf, 'logistic_regression': accuracy_logreg, 'engression': accuracy_engression, 'GAM': accuracy_gam, 'GP': accuracy_GP} 

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(accuracy_results.items()), columns=['Method', 'Accuracy'])

    # Create the directory if it doesn't exist
    os.makedirs('RESULTS/K_MEDOIDS', exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f'RESULTS/K_MEDOIDS/{task_id}_kmedoids_accuracy_results.csv', index=False)