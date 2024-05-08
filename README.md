# ML models for extrapolation
This repository contains the code to run large scale experiments, and evaluate the extrapolation performance of several machine learning models. This is my MSc thesis project at ETH Zurich. The code is organised as follows:

- There is a folder for datasets containing only numerical features, and a folder for datasets with both numerical and categorical features.
- Within them, there is a folder for regression tasks, and a folder for classification tasks.
- Within them, there is a folder for each metric of interest (RMSE and CRPS for regression, accuracy and logloss for classification).
- In these folders, there is a script for each technique to define the extrapolation set. This script runs the experiments on all datasets, and generate the results.
- Separate scripts are created for GAM, Gaussian Process (which is anyway not included in the analysis for now), and sometimes FT-Transformer. This is because these methods sometimes fail, and I wanted to be able to run them separately.
- The notebooks "analysis_normalized_accuracy", "analysis_ranks", and "analysis_relative_differences" create the plots for these 3 metrics, aggregating across datasets.
- The R scripts "create_table" create the tables of results.
- Please note that the plots and tables averaging over both datasets with only numerical features, and datasets with both numerical and categorical features (as presented in the paper) are generated in the folder "ONLY NUMERICAL FEATURES"
- The notebooks "analysis_y", which are located in the folders RMSE, generate the histograms of y.
