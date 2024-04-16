import pandas as pd
import numpy as np
import openml
from sklearn.linear_model import LogisticRegression 
import lightgbm as lgbm
import optuna
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import Matern
from engression import engression
import torch
from scipy.spatial.distance import mahalanobis
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
import random
import os
from pygam import LogisticGAM
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
from utils import EarlyStopping, train, train_trans, train_no_early_stopping, train_trans_no_early_stopping
from torch.utils.data import TensorDataset, DataLoader
import re
import shutil
import gpboost as gpb

#SUITE_ID = 336 # Regression on numerical features
SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

task_id=361055

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

print(f"Task {task_id}")

# Load the existing DataFrame
df = pd.read_csv(f'ONLY NUMERICAL FEATURES/CLASSIFICATION/ACCURACY/RESULTS/MAHALANOBIS/{task_id}_mahalanobis_accuracy_results.csv')