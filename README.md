# ML models for extrapolation
This repository contains the code to run large scale experiments, and evaluate the extrapolation performance of several machine learning models. This is my MSc thesis project at ETH Zurich. The code is organised as follows:

- There is a folder for datasets containing only numerical features, and a folder for datasets with both numerical and categorical features
- Within them, there is a folder for regression tasks, and a folder for classification tasks
- Within them, there is a folder for each metric of interest (RMSE and CRPS for regression, accuracy and logloss for classification)
- In these folders, there is a script for each technique to define the extrapolation set. This script runs the experiments on all datasets, and generate the results.
- Separate scripts are created for GAM, Gaussian Process (which is anyway not included in the analysis for now), and sometimes FT-Transformer. This is because these methods sometimes fail, and I wanted to be able to run them separately
- 
