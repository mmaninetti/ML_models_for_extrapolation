
import pandas as pd
import numpy as np
import openml
from sklearn.linear_model import LinearRegression 
import lightgbm as lgbm
import optuna
from scipy.spatial.distance import mahalanobis
from sklearn.gaussian_process import GaussianProcessRegressor
import gpytorch
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern
from engression import engression, engression_bagged
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from rtdl_revisiting_models import MLP, ResNet, FTTransformer


#openml.config.apikey = 'FILL_IN_OPENML_API_KEY'  # set the OpenML Api Key
SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

task = openml.tasks.get_task(361072)  # download the OpenML task
dataset = task.get_dataset()

X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute)


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


X_train_tensor = torch.tensor(X_train_.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_.values, dtype=torch.float32)


X_train_tensor.shape
y_train_tensor.shape


seed=10
 
# #### MLP


N_TRIALS=5

d_out = 1  
d_in=X_train_.shape[1]

def MLP_opt(trial):

    seed=10
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
    n_epochs=trial.suggest_int('n_epochs', 100, 5000)
    learning_rate=trial.suggest_float('learning_rate', 0.0001, 0.05, log=True)
    optimizer=torch.optim.Adam(MLP_model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    loss_Adam=[]

    for i in range(n_epochs):
        # making a pridiction in forward pass
        y_train_hat = MLP_model(torch.Tensor(X_train_.values)).reshape(-1,)
        # calculating the loss between original and predicted data points
        loss = criterion(y_train_hat, torch.Tensor(y_train_.values))
        # store loss into list
        loss_Adam.append(loss.item())
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()

    y_val_hat_MLP=(MLP_model(torch.Tensor(X_val.values)).reshape(-1,)).detach().numpy()
    RMSE_MLP=np.sqrt(np.mean((y_val-y_val_hat_MLP)**2))

    return RMSE_MLP

sampler_MLP = optuna.samplers.TPESampler(seed=10)
study_MLP = optuna.create_study(sampler=sampler_MLP, direction='minimize')
study_MLP.optimize(MLP_opt, n_trials=N_TRIALS)


seed=10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

MLP_model = MLP(
    d_in=d_in,
    d_out=d_out,
    n_blocks=study_MLP.best_params['n_blocks'],
    d_block=study_MLP.best_params['d_block'],
    dropout=study_MLP.best_params['dropout'],
    )
n_epochs=study_MLP.best_params['n_epochs']
learning_rate=study_MLP.best_params['learning_rate']
optimizer=torch.optim.Adam(MLP_model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
loss_Adam=[]

for i in range(n_epochs):
    # making a pridiction in forward pass
    y_train_hat = MLP_model(torch.Tensor(X_train.values)).reshape(-1,)
    # calculating the loss between original and predicted data points
    loss = criterion(y_train_hat, torch.Tensor(y_train.values))
    # store loss into list
    loss_Adam.append(loss.item())
    # zeroing gradients after each iteration
    optimizer.zero_grad()
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updating the parameters after each iteration
    optimizer.step()

y_test_hat_MLP=(MLP_model(torch.Tensor(X_test.values)).reshape(-1,)).detach().numpy()
RMSE_MLP=np.sqrt(np.mean((y_test-y_test_hat_MLP)**2))
print("RMSE MLP: ", RMSE_MLP)

 
# #### ResNet


N_TRIALS=5

d_out = 1  
d_in=X_train_.shape[1]

def ResNet_opt(trial):

    seed=10
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
    n_epochs=trial.suggest_int('n_epochs', 100, 5000)
    learning_rate=trial.suggest_float('learning_rate', 0.0001, 0.05, log=True)
    optimizer=torch.optim.Adam(ResNet_model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    loss_Adam=[]

    for i in range(n_epochs):
        # making a pridiction in forward pass
        y_train_hat = ResNet_model(torch.Tensor(X_train_.values)).reshape(-1,)
        # calculating the loss between original and predicted data points
        loss = criterion(y_train_hat, torch.Tensor(y_train_.values))
        # store loss into list
        loss_Adam.append(loss.item())
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()

    y_val_hat_ResNet=(ResNet_model(torch.Tensor(X_val.values)).reshape(-1,)).detach().numpy()
    RMSE_ResNet=np.sqrt(np.mean((y_val-y_val_hat_ResNet)**2))

    return RMSE_ResNet

sampler_ResNet = optuna.samplers.TPESampler(seed=10)
study_ResNet = optuna.create_study(sampler=sampler_ResNet, direction='minimize')
study_ResNet.optimize(ResNet_opt, n_trials=N_TRIALS)


seed=10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

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
n_epochs=study_ResNet.best_params['n_epochs']
learning_rate=study_ResNet.best_params['learning_rate']
optimizer=torch.optim.Adam(ResNet_model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
loss_Adam=[]

for i in range(n_epochs):
    # making a pridiction in forward pass
    y_train_hat = ResNet_model(torch.Tensor(X_train.values)).reshape(-1,)
    # calculating the loss between original and predicted data points
    loss = criterion(y_train_hat, torch.Tensor(y_train.values))
    # store loss into list
    loss_Adam.append(loss.item())
    # zeroing gradients after each iteration
    optimizer.zero_grad()
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updating the parameters after each iteration
    optimizer.step()

y_test_hat_ResNet=(ResNet_model(torch.Tensor(X_test.values)).reshape(-1,)).detach().numpy()
RMSE_ResNet=np.sqrt(np.mean((y_test-y_test_hat_ResNet)**2))
print("RMSE ResNet: ", RMSE_ResNet)

 
# #### FFTransformer


N_TRIALS=5

d_out = 1  
d_in=X_train_.shape[1]

def FTTrans_opt(trial):

    seed=10
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

    n_epochs=trial.suggest_int('n_epochs', 100, 5000)
    learning_rate=trial.suggest_float('learning_rate', 0.0001, 0.05, log=True)
    optimizer=torch.optim.Adam(FTTrans_model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    loss_Adam=[]

    for i in range(n_epochs):
        # making a pridiction in forward pass
        y_train_hat = FTTrans_model(torch.Tensor(X_train_.values),None).reshape(-1,)
        # calculating the loss between original and predicted data points
        loss = criterion(y_train_hat, torch.Tensor(y_train_.values))
        # store loss into list
        loss_Adam.append(loss.item())
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()

    y_val_hat_FTTrans=(FTTrans_model(torch.Tensor(X_val.values), None).reshape(-1,)).detach().numpy()
    RMSE_FTTrans=np.sqrt(np.mean((y_val-y_val_hat_FTTrans)**2))

    return RMSE_FTTrans

sampler_FTTrans = optuna.samplers.TPESampler(seed=10)
study_FTTrans = optuna.create_study(sampler=sampler_FTTrans, direction='minimize')
study_FTTrans.optimize(FTTrans_opt, n_trials=N_TRIALS)


seed=10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

FTTrans_model = FTTransformer(
    n_cont_features=d_in,
    cat_cardinalities=[],
    d_out=d_out,
    n_blocks=study_FTTrans.best_params['n_blocks'],
    d_block=study_FTTrans.best_params['d_block'],
    attention_n_heads=study_FTTrans.best_params['attention_n_heads'],
    attention_dropout=study_FTTrans.best_params['attention_dropout'],
    ffn_d_hidden=None,
    ffn_d_hidden_multiplier=study_FTTrans.best_params['ffn_d_hidden_multiplier'],
    ffn_dropout=study_FTTrans.best_params['ffn_dropout'],
    residual_dropout=study_FTTrans.best_params['residual_dropout'],
    )
n_epochs=study_FTTrans.best_params['n_epochs']
learning_rate=study_FTTrans.best_params['learning_rate']
optimizer=torch.optim.Adam(FTTrans_model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
loss_Adam=[]

for i in range(n_epochs):
    # making a pridiction in forward pass
    y_train_hat = FTTrans_model(torch.Tensor(X_train.values), None).reshape(-1,)
    # calculating the loss between original and predicted data points
    loss = criterion(y_train_hat, torch.Tensor(y_train.values))
    # store loss into list
    loss_Adam.append(loss.item())
    # zeroing gradients after each iteration
    optimizer.zero_grad()
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updating the parameters after each iteration
    optimizer.step()

y_test_hat_FTTrans=(FTTrans_model(torch.Tensor(X_test.values)).reshape(-1,)).detach().numpy()
RMSE_FTTrans=np.sqrt(np.mean((y_test-y_test_hat_FTTrans)**2))
print("RMSE FTTrans: ", RMSE_FTTrans)

 
# #### Boosted trees, random forest, engression, linear regression


N_TRIALS=5

def boosted(trial):

    params = {'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
              'n_estimators': trial.suggest_int('n_estimators', 100, 500),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
              'max_depth': trial.suggest_int('max_depth', 1, 30),
              'min_child_samples': trial.suggest_int('min_child_samples', 10, 100)}
    
    boosted_tree_model=lgbm.LGBMRegressor(**params)
    boosted_tree_model.fit(X_train_, y_train_)
    y_val_hat_boost=boosted_tree_model.predict(X_val)
    RMSE_boost=np.sqrt(np.mean((y_val-y_val_hat_boost)**2))

    return RMSE_boost

sampler_boost = optuna.samplers.TPESampler(seed=10)
study_boost = optuna.create_study(sampler=sampler_boost, direction='minimize')
study_boost.optimize(boosted, n_trials=N_TRIALS)

boosted_model=lgbm.LGBMRegressor(**study_boost.best_params)

def rf(trial):

    params = {'n_estimators': trial.suggest_int('n_estimators', 100, 500),
              'max_depth': trial.suggest_int('max_depth', 1, 30),
              'max_features': trial.suggest_int('max_features', 1, 30),
              'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100)}
    
    rf_model=RandomForestRegressor(**params)
    rf_model.fit(X_train_, y_train_)
    y_val_hat_rf=rf_model.predict(X_val)
    RMSE_rf=np.sqrt(np.mean((y_val-y_val_hat_rf)**2))

    return RMSE_rf

sampler_rf = optuna.samplers.TPESampler(seed=10)
study_rf = optuna.create_study(sampler=sampler_rf, direction='minimize')
study_rf.optimize(rf, n_trials=N_TRIALS)

rf_model=RandomForestRegressor(**study_rf.best_params)


def engressor_NN(trial):

    params = {'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
              'num_epoches': trial.suggest_int('num_epoches', 100, 1000),
              'num_layer': trial.suggest_int('num_layer', 2, 5),
              'hidden_dim': trial.suggest_int('hidden_dim', 50, 100),
              'noise_dim': trial.suggest_int('noise_dim', 50, 100),}
    
    engressor_model=engression(torch.Tensor(np.array(X_train_)), torch.Tensor(np.array(y_train_).reshape(-1,1)), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=1000)
    y_val_hat_engression=engressor_model.predict(torch.Tensor(np.array(X_val)), target="mean")
    RMSE_engression=np.sqrt((((torch.Tensor(np.array(y_val).reshape(-1,1)))-y_val_hat_engression)**2).mean(axis=0))

    return RMSE_engression

sampler_engression = optuna.samplers.TPESampler(seed=10)
study_engression = optuna.create_study(sampler=sampler_engression, direction='minimize')
study_engression.optimize(engressor_NN, n_trials=N_TRIALS)


boosted_model.fit(X_train, y_train)
y_test_hat_boosted=boosted_model.predict(X_test)
RMSE_boosted=np.sqrt(np.mean((y_test-y_test_hat_boosted)**2))

rf_model.fit(X_train, y_train)
y_test_hat_rf=rf_model.predict(X_test)
RMSE_rf=np.sqrt(np.mean((y_test-y_test_hat_rf)**2))

lin_reg=LinearRegression()
lin_reg.fit(X_train, y_train)
y_test_hat_linreg=lin_reg.predict(X_test)
RMSE_linreg=np.sqrt(np.mean((y_test-y_test_hat_linreg)**2))

params=study_engression.best_params
engressor_model=engression(torch.Tensor(np.array(X_train)), torch.Tensor(np.array(y_train).reshape(-1,1)), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=1000)
y_test_hat_engression=engressor_model.predict(torch.Tensor(np.array(X_test)), target="mean")
RMSE_engression=np.sqrt((((torch.Tensor(np.array(y_test).reshape(-1,1)))-y_test_hat_engression)**2).mean(axis=0))

print("RMSE linear regression: ",RMSE_linreg)
print("RMSE boosted trees", RMSE_boosted)
print("RMSE random forest", RMSE_rf)
print("RMSE engression", RMSE_engression)










 
# #### Extra stuff

 
# #### Old way to define MLP

 
# N_TRIALS=5
# 
# def MLP_opt(trial):
# 
#     seed=10
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
# 
#     n_layers = trial.suggest_int("n_layers", 1, 5)
#     layers = []
# 
#     in_features = X_train_.shape[1]
#     for i in range(n_layers):
#         out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
#         layers.append(nn.Linear(in_features, out_features))
#         layers.append(nn.ReLU())
#         p = trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)
#         layers.append(nn.Dropout(p))
# 
#         in_features = out_features
#     layers.append(nn.Linear(in_features, 1))
#     MLP_model=nn.Sequential(*layers)
#     n_epochs=trial.suggest_int('n_epochs', 100, 5000)
#     learning_rate=trial.suggest_float('learning_rate', 0.0001, 0.05)
#     optimizer=torch.optim.Adam(MLP_model.parameters(), lr=learning_rate)
#     criterion = torch.nn.MSELoss()
#     loss_Adam=[]
# 
#     for i in range(n_epochs):
#         # making a pridiction in forward pass
#         y_train_hat = MLP_model(torch.Tensor(X_train_.values)).reshape(-1,)
#         # calculating the loss between original and predicted data points
#         loss = criterion(y_train_hat, torch.Tensor(y_train_.values))
#         # store loss into list
#         loss_Adam.append(loss.item())
#         # zeroing gradients after each iteration
#         optimizer.zero_grad()
#         # backward pass for computing the gradients of the loss w.r.t to learnable parameters
#         loss.backward()
#         # updating the parameters after each iteration
#         optimizer.step()
# 
#     y_val_hat_MLP=(MLP_model(torch.Tensor(X_val.values)).reshape(-1,)).detach().numpy()
#     RMSE_MLP=np.sqrt(np.mean((y_val-y_val_hat_MLP)**2))
# 
#     return RMSE_MLP
# 
# sampler_MLP = optuna.samplers.TPESampler(seed=10)
# study_MLP = optuna.create_study(sampler=sampler_MLP, direction='minimize')
# study_MLP.optimize(MLP_opt, n_trials=N_TRIALS)
# 
# #gp_model=GaussianProcessRegressor(kernel=Matern(length_scale=study_gp.best_params['lenghtscale'], nu=1.5))


seed=10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

layers = []

in_features = X_train.shape[1]
for i in range(study_MLP.best_params['n_layers']):
    out_features = study_MLP.best_params["n_units_l{}".format(i)]
    layers.append(nn.Linear(in_features, out_features))
    layers.append(nn.ReLU())
    p = study_MLP.best_params["dropout_l{}".format(i)]
    layers.append(nn.Dropout(p))

    in_features = out_features

layers.append(nn.Linear(in_features, 1))
MLP_model=nn.Sequential(*layers)
n_epochs=study_MLP.best_params['n_epochs']
learning_rate=study_MLP.best_params['learning_rate']
optimizer=torch.optim.Adam(MLP_model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
loss_Adam=[]

for i in range(n_epochs):
    # making a pridiction in forward pass
    y_train_hat = MLP_model(torch.Tensor(X_train.values)).reshape(-1,)
    # calculating the loss between original and predicted data points
    loss = criterion(y_train_hat, torch.Tensor(y_train.values))
    # store loss into list
    loss_Adam.append(loss.item())
    # zeroing gradients after each iteration
    optimizer.zero_grad()
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updating the parameters after each iteration
    optimizer.step()

y_test_hat_MLP=(MLP_model(torch.Tensor(X_test.values)).reshape(-1,)).detach().numpy()
RMSE_MLP=np.sqrt(np.mean((y_test-y_test_hat_MLP)**2))
print("RMSE MLP: ", RMSE_MLP)

