
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
    X_train__tensor = X_train__tensor.cuda()
    y_train__tensor = y_train__tensor.cuda()
    X_train_tensor = X_train_tensor.cuda()
    y_train_tensor = y_train_tensor.cuda()
    X_val_tensor = X_val_tensor.cuda()
    y_val_tensor = y_val_tensor.cuda()
    X_test_tensor = X_test_tensor.cuda()
    y_test_tensor = y_test_tensor.cuda()
print(X_train__tensor.device)

# Create flattened versions of the data
y_val_np = y_val.values.flatten()
y_test_np = y_test.values.flatten()

#### Gaussian process
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Define the learning params
training_iterations = GP_ITERATIONS

# Define the kernels
kernels = [
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=X_train_.shape[1])),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=X_train_.shape[1])),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X_train_.shape[1])),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X_train_.shape[1])),
]

best_RMSE = float('inf')
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
    train_GP(model,X_train__tensor,y_train__tensor,training_iterations,mll,optimizer)
    
    # Set the model in evaluation mode
    model.eval()
    likelihood.eval()

    # Make predictions on the validation set
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_pred = model(X_val_tensor)

    # Calculate RMSE
    RMSE = torch.sqrt(torch.mean(torch.square(y_val_tensor - y_pred.mean)))

    # Update the best kernel if the current kernel has a lower RMSE
    if RMSE < best_RMSE:
        best_RMSE = RMSE
        best_kernel = kernel


# Set the random seed for reproducibility

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = best_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Define the learning params
training_iterations = GP_ITERATIONS

# Initialize the Gaussian Process model and likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_train_tensor, y_train_tensor, likelihood)

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

if torch.cuda.is_available():
    model = model.cuda()

# Train the model
train_GP(model,X_train_tensor,y_train_tensor,training_iterations,mll,optimizer)

# Set the model in evaluation mode
model.eval()
likelihood.eval()

# Make predictions on the validation set
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    y_pred = model(X_test_tensor)

# Calculate RMSE
RMSE_GP = torch.sqrt(torch.mean(torch.square(y_test_tensor - y_pred.mean)))
print("RMSE GP: ", RMSE_GP)


#### MLP
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
    loss_Adam=[]

    if torch.cuda.is_available():
        MLP_model = MLP_model.cuda()

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=False)
    n_epochs=train(MLP_model, criterion, loss_Adam, optimizer, n_epochs, X_train__tensor, y_train__tensor, X_val_tensor, y_val_tensor, early_stopping)
    n_epochs = trial.suggest_int('n_epochs', n_epochs, n_epochs)

    # Point prediction
    y_val_hat_MLP = (MLP_model(X_val_tensor).reshape(-1,))
    RMSE_MLP=torch.sqrt(torch.mean(torch.square(y_val_tensor - y_val_hat_MLP)))

    return RMSE_MLP

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
loss_Adam=[]

train_no_early_stopping(MLP_model, criterion, loss_Adam, optimizer, n_epochs, X_train_tensor, y_train_tensor)

# Point prediction
y_test_hat_MLP = (MLP_model(X_test_tensor).reshape(-1,))
RMSE_MLP=torch.sqrt(torch.mean(torch.square(y_test_tensor - y_test_hat_MLP)))
print("RMSE MLP: ", RMSE_MLP)

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
    loss_Adam=[]

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=False)
    n_epochs=train(ResNet_model, criterion, loss_Adam, optimizer, n_epochs, X_train__tensor, y_train__tensor, X_val_tensor, y_val_tensor, early_stopping)
    n_epochs = trial.suggest_int('n_epochs', n_epochs, n_epochs)

    # Point prediction
    y_val_hat_ResNet = (ResNet_model(X_val_tensor).reshape(-1,))
    RMSE_ResNet=torch.sqrt(torch.mean(torch.square(y_val_tensor - y_val_hat_ResNet)))

    return RMSE_ResNet

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
loss_Adam=[]

train_no_early_stopping(ResNet_model, criterion, loss_Adam, optimizer, n_epochs, X_train_tensor, y_train_tensor)

# Point prediction
y_test_hat_ResNet = (ResNet_model(X_test_tensor).reshape(-1,))
RMSE_ResNet=torch.sqrt(torch.mean(torch.square(y_test_tensor - y_test_hat_ResNet)))
print("RMSE ResNet: ", RMSE_ResNet)

#### FFTransformer
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
    loss_Adam=[]

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=False)
    n_epochs=train_trans(FTTrans_model, criterion, loss_Adam, optimizer, n_epochs, X_train__tensor, y_train__tensor, X_val_tensor, y_val_tensor, early_stopping)
    n_epochs = trial.suggest_int('n_epochs', n_epochs, n_epochs)

    # Point prediction
    y_val_hat_FTTrans = (FTTrans_model(X_val_tensor, None).reshape(-1,))
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

# #### Boosted trees, random forest, engression, linear regression

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

sampler_boost = optuna.samplers.TPESampler(seed=seed)
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

sampler_rf = optuna.samplers.TPESampler(seed=seed)
study_rf = optuna.create_study(sampler=sampler_rf, direction='minimize')
study_rf.optimize(rf, n_trials=N_TRIALS)
rf_model=RandomForestRegressor(**study_rf.best_params)


def engressor_NN(trial):

    params = {'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
              'num_epoches': trial.suggest_int('num_epoches', 100, 1000),
              'num_layer': trial.suggest_int('num_layer', 2, 5),
              'hidden_dim': trial.suggest_int('hidden_dim', 100, 500),}
    params['noise_dim']=params['hidden_dim']

    # Check if CUDA is available and if so, move the tensors and the model to the GPU
    if torch.cuda.is_available():
        engressor_model=engression(X_train__tensor, y_train__tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=1000, device="cuda")
    else: 
        engressor_model=engression(X_train__tensor, y_train__tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=1000)
    
    # Generate a sample from the engression model for each data point
    y_val_hat_engression=engressor_model.predict(X_val_tensor, target="mean")
    RMSE_engression=torch.sqrt(torch.mean(torch.square(y_val_tensor.reshape(-1,1) - y_val_hat_engression)))

    return RMSE_engression

sampler_engression = optuna.samplers.TPESampler(seed=seed)
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
params['noise_dim']=params['hidden_dim']
# Check if CUDA is available and if so, move the tensors and the model to the GPU
if torch.cuda.is_available():
    engressor_model=engression(X_train_tensor, y_train_tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=1000, device="cuda")
else: 
    engressor_model=engression(X_train_tensor, y_train_tensor.reshape(-1,1), lr=params['learning_rate'], num_epoches=params['num_epoches'],num_layer=params['num_layer'], hidden_dim=params['hidden_dim'], noise_dim=params['noise_dim'], batch_size=1000)
y_test_hat_engression=engressor_model.predict(X_test_tensor, target="mean")
RMSE_engression=torch.sqrt(torch.mean(torch.square(y_test_tensor.reshape(-1,1) - y_test_hat_engression)))


print("RMSE linear regression: ",RMSE_linreg)
print("RMSE boosted trees", RMSE_boosted)
print("RMSE random forest", RMSE_rf)
print("RMSE engression", RMSE_engression)

#### GAM model
def gam_model(trial):

    # Define the hyperparameters to optimize
    params = {'n_splines': trial.suggest_int('n_splines', 5, 20),
              'lam': trial.suggest_loguniform('lam', 1e-3, 1)}

    # Create and train the model
    gam = LinearGAM(s(0, n_splines=params['n_splines'], lam=params['lam'])).fit(X_train_, y_train_)

    # Predict on the validation set and calculate the RMSE
    y_val_hat_gam = gam.predict(X_val)
    RMSE_gam = np.sqrt(np.mean((y_val - y_val_hat_gam) ** 2))

    return RMSE_gam

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
# Calculate the RMSE
RMSE_gam = np.sqrt(np.mean((y_test - y_test_hat_gam) ** 2))
print("RMSE GAM: ", RMSE_gam)

RMSE_results = {'GP': RMSE_GP, 'MLP': RMSE_MLP, 'ResNet': RMSE_ResNet, 'FTTrans': RMSE_FTTrans, 'boosted_trees': RMSE_boosted, 'rf': RMSE_rf, 'linear_regression': RMSE_linreg, 'engression': RMSE_engression, 'GAM': RMSE_gam}  

# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(RMSE_results.items()), columns=['Method', 'RMSE'])

# Create the directory if it doesn't exist
os.makedirs('RESULTS/MAHALANOBIS', exist_ok=True)

# Save the DataFrame to a CSV file
df.to_csv(f'RESULTS/MAHALANOBIS/{task_id}_mahalanobis_RMSE_results.csv', index=False)