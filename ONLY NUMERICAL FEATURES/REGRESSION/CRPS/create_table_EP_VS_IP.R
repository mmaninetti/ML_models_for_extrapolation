#######################
### Aggregate results of different experiments
### Author: Fabio Sigrist
### Date: June 2023
#######################

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
library(tidyverse)
library(xtable)
library(ggplot2)
library(scales)


library(plyr)
library(ggplot2)


list_directories <- c("RESULTS/CLUSTERING", "RESULTS/UMAP_DECOMPOSITION", "RESULTS/SPATIAL_DEPTH", "RESULTS/MAHALANOBIS", "RESULTS/GOWER", "RESULTS/K_MEDOIDS", "RESULTS/UMAP_DECOMPOSITION_CAT")
random_directory <- "RESULTS/RANDOM"
methods <- c('constant', 'MLP', 'ResNet', 'FTTrans', 'distributional_boosted_trees', 'drf', 'boosted_trees', 'rf', 'linear_regression', 'engression')

#### Define the unique task_ids
task_ids <- c()
for (directory in list_directories) {
  filenames <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
  for (filename in filenames) {
    task_id <- sub("^(\\d+)_.*", "\\1", basename(filename))
    task_ids <- c(task_ids, task_id)
  }
}
task_ids <- unique(task_ids)


# Create an empty data frame to store the results
results_agg <- data.frame(task_id = character(), stringsAsFactors = FALSE)

# Add a numeric column for each method
for (method in methods) {
  results_agg[[method]] <- numeric()
}

result_CRPS_all <- data.frame(method = methods, stringsAsFactors = FALSE)
for (task_id in task_ids)
{
  # Create a row for the current task_id
  result_row <- data.frame(task_id = task_id, stringsAsFactors = FALSE)
  
  # Add a column for each method
  for (method in methods) {
    result_row[[method]] <- NA
  }

  # Create a dataset for the CRPS with the correct number of rows
  result_CRPS <- data.frame(method = methods, stringsAsFactors = FALSE)
  
  # Iterate through the 4 directories
  for (directory in list_directories)
  {
    if (directory %in% c("RESULTS/CLUSTERING", "RESULTS/UMAP_DECOMPOSITION", "RESULTS/SPATIAL_DEPTH", "RESULTS/MAHALANOBIS"))
    {
      random_directory <- "RESULTS/RANDOM"
      i=1
    }
    else
    {
      random_directory <- "RESULTS/RANDOM_CAT"
      i=2
    }
    # Construct the filename for the current task_id and directory
    if (directory=="RESULTS/CLUSTERING")
    {
      filename <- file.path(directory, paste0(task_id, "_clustering_CRPS_results.csv"))
    }
    if (directory=="RESULTS/UMAP_DECOMPOSITION" | directory=="RESULTS/UMAP_DECOMPOSITION_CAT")
    {
      filename <- file.path(directory, paste0(task_id, "_umap_decomposition_CRPS_results.csv"))
    }
    if (directory=="RESULTS/SPATIAL_DEPTH")
    {
      filename <- file.path(directory, paste0(task_id, "_spatial_depth_CRPS_results.csv"))
    }
    if (directory=="RESULTS/MAHALANOBIS")
    {
      filename <- file.path(directory, paste0(task_id, "_mahalanobis_CRPS_results.csv"))
    }
    if (directory=="RESULTS/GOWER")
    {
      filename <- file.path(directory, paste0(task_id, "_gower_CRPS_results.csv"))
    }
    if (directory=="RESULTS/K_MEDOIDS")
    {
      filename <- file.path(directory, paste0(task_id, "_k_medoids_CRPS_results.csv"))
    }
    
    
    # Check if the file exists
    if (file.exists(filename))
    {
      # Load the results dataset
      results_dataset <- head(read.csv(filename),-2)
      
      # Extract the Method and CRPS columns
      method <- results_dataset$Method
      CRPS <- results_dataset$CRPS
      
      # Load the random file
      random_filename <- file.path(random_directory, paste0(task_id, "_random_crps_results.csv"))
      random_dataset <- head(read.csv(random_filename),-i)
      
      # Extract the Method and CRPS columns
      random_method <- random_dataset$Method
      random_CRPS <- random_dataset$CRPS
      
      diff <- 100*(CRPS - random_CRPS) / random_CRPS

      # Append the Method and CRPS to the result_row
      result_CRPS <- cbind(result_CRPS, diff)
      result_CRPS_all <- cbind(result_CRPS_all, diff)
    }
  }
  
  # Calculate the mean of the CRPS for each method
  if (ncol(result_CRPS) > 2) 
  {
    result_CRPS_mean <- rowMeans(result_CRPS[, -1], na.rm=TRUE)
  }
  else
  {
    result_CRPS_mean <- result_CRPS$diff
  }

  # Append the result_CRPS_mean to result_row
  result_row[, methods] <- result_CRPS_mean
  
  # Append the result_row to the results_agg data frame
  results_agg <- rbind(results_agg, result_row)
}

# Reorder the columns in results_agg
results_agg <- results_agg[,c("task_id", "constant", "linear_regression", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]


# Print the new dataset with the Method and CRPS columns
print(results_agg)
results<-results_agg


# Change names
models <- methods
models_new_name <- c('const.', 'lin. reg.', 'RF', 'GBT', 'engression', 'MLP', 'ResNet', 'FT-Trans.', 'DRF', 'DGBT')
# Change names
colnames(results) <- c("task_id", models_new_name)

rowdiffs <- rowMeans(result_CRPS_all[,-1], na.rm=TRUE)
avg_diff <- data.frame(task_id="Avg. diff",const=rowdiffs[1], 'lin. reg.'=rowdiffs[9], RF=rowdiffs[8],GBT=rowdiffs[7], engression=rowdiffs[10], MLP=rowdiffs[2], ResNet=rowdiffs[3], FT=rowdiffs[4], DRF=rowdiffs[6], DGBT=rowdiffs[5])
colnames(avg_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_diff)

rowdiffs <- apply(result_CRPS_all[,-1], 1, median, na.rm=T)
avg_diff <- data.frame(task_id="Median diff",const=rowdiffs[1], 'lin. reg.'=rowdiffs[9], RF=rowdiffs[8],GBT=rowdiffs[7], engression=rowdiffs[10], MLP=rowdiffs[2], ResNet=rowdiffs[3], FT=rowdiffs[4], DRF=rowdiffs[6], DGBT=rowdiffs[5])
colnames(avg_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_diff)

results$"Avg diff"<-rowMeans(results[,-1], na.rm=TRUE)

## Create table
num_digits <- 3
output <- results
# Set the number of significant digits
output[, -1] <- signif(output[, -1], num_digits)

# Find the lowest value in each row
lowest_values <- apply(output[, -1], 1, function(x) min(x, na.rm=TRUE))

# Find the highest value in the second-to-last column
#highest_value <- max(output[nrow(output) - 1, -1], na.rm=TRUE)

# Convert numbers smaller than 0.1 and bigger than 100 to scientific notation
output[, -1][((abs(output[, -1]) < 0.1) | ((abs(output[, -1])) >=1000)) & 0==is.na(output[, -1])] <- format(output[, -1][((abs(output[, -1]) < 0.1) | ((abs(output[, -1])) >=1000)) & 0==is.na(output[, -1])], scientific = TRUE)
lowest_values[((abs(lowest_values) < 0.1) | ((abs(lowest_values)) >=1000)) & 0==is.na(lowest_values)] <- format(lowest_values[((abs(lowest_values) < 0.1) | ((abs(lowest_values)) >=1000)) & 0==is.na(lowest_values)], scientific=TRUE)

# Loop through each row and format the lowest value and highest value in bold
for (i in 1:nrow(output)) {
  output[i, -1] <- ifelse(output[i, -1] == lowest_values[i], paste0("\\textbf{", output[i, -1], "}"), output[i, -1])
}
#output[nrow(output) - 1, -1] <- ifelse(output[nrow(output) - 1, -1] == highest_value, paste0("\\textbf{", output[nrow(output) - 1, -1], "}"), output[nrow(output) - 1, -1])

caption <- paste0("Average relative difference of test CRPS using EP VS using IP, in \\%. 
                  Best results are bold. 
                  'Avg. diff.' denotes the average of each column.
                  'Median diff.' denotes the median of each column.")
                   #Bold results are non-inferior to the best result in a paired t-test at a 5\\% level.
label <- paste0("TABLES/table_results_CRPS_only_num_EP_VS_IP")
filename <- paste0(label, ".tex", sep="")
tab <- xtable(output, caption = caption, label = label)

print(tab, file = filename, sanitize.text.function = function(str) gsub("_", "\\_", str, fixed = TRUE),
      table.placement = getOption("xtable.table.placement", "ht!"), include.rownames = FALSE, hline.after = c(-1, -1, 0, (length(task_ids)), rep(dim(output)[1], 2)),
      size = "\\footnotesize", caption.placement = "bottom") #, floating.environment = "sidewaystable"




##############################################
########## ONLY CLUSTERING ##################
##############################################
list_directories <- c("RESULTS/CLUSTERING")
random_directory <- "RESULTS/RANDOM"
methods <- c('constant', 'MLP', 'ResNet', 'FTTrans', 'distributional_boosted_trees', 'drf', 'boosted_trees', 'rf', 'linear_regression', 'engression')

#### Define the unique task_ids
task_ids <- c()
for (directory in list_directories) {
  filenames <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
  for (filename in filenames) {
    task_id <- sub("^(\\d+)_.*", "\\1", basename(filename))
    task_ids <- c(task_ids, task_id)
  }
}
task_ids <- unique(task_ids)


# Create an empty data frame to store the results
results_agg <- data.frame(task_id = character(), stringsAsFactors = FALSE)

# Add a numeric column for each method
for (method in methods) {
  results_agg[[method]] <- numeric()
}

result_CRPS_all <- data.frame(method = methods, stringsAsFactors = FALSE)
for (task_id in task_ids)
{
  # Create a row for the current task_id
  result_row <- data.frame(task_id = task_id, stringsAsFactors = FALSE)
  
  # Add a column for each method
  for (method in methods) {
    result_row[[method]] <- NA
  }
  
  # Create a dataset for the CRPS with the correct number of rows
  result_CRPS <- data.frame(method = methods, stringsAsFactors = FALSE)
  
  # Iterate through the 4 directories
  for (directory in list_directories)
  {
    if (directory %in% c("RESULTS/CLUSTERING", "RESULTS/UMAP_DECOMPOSITION", "RESULTS/SPATIAL_DEPTH", "RESULTS/MAHALANOBIS"))
    {
      random_directory <- "RESULTS/RANDOM"
      i = 1
    }
    else
    {
      random_directory <- "RESULTS/RANDOM_CAT"
      i = 2
    }
    # Construct the filename for the current task_id and directory
    if (directory=="RESULTS/CLUSTERING")
    {
      filename <- file.path(directory, paste0(task_id, "_clustering_CRPS_results.csv"))
    }
    if (directory=="RESULTS/UMAP_DECOMPOSITION" | directory=="RESULTS/UMAP_DECOMPOSITION_CAT")
    {
      filename <- file.path(directory, paste0(task_id, "_umap_decomposition_CRPS_results.csv"))
    }
    if (directory=="RESULTS/SPATIAL_DEPTH")
    {
      filename <- file.path(directory, paste0(task_id, "_spatial_depth_CRPS_results.csv"))
    }
    if (directory=="RESULTS/MAHALANOBIS")
    {
      filename <- file.path(directory, paste0(task_id, "_mahalanobis_CRPS_results.csv"))
    }
    if (directory=="RESULTS/GOWER")
    {
      filename <- file.path(directory, paste0(task_id, "_gower_CRPS_results.csv"))
    }
    if (directory=="RESULTS/K_MEDOIDS")
    {
      filename <- file.path(directory, paste0(task_id, "_k_medoids_CRPS_results.csv"))
    }
    
    
    # Check if the file exists
    if (file.exists(filename))
    {
      # Load the results dataset
      results_dataset <- head(read.csv(filename),-2)
      
      # Extract the Method and CRPS columns
      method <- results_dataset$Method
      CRPS <- results_dataset$CRPS
      
      # Load the random file
      random_filename <- file.path(random_directory, paste0(task_id, "_random_crps_results.csv"))
      random_dataset <- head(read.csv(random_filename),-i)
      
      # Extract the Method and CRPS columns
      random_method <- random_dataset$Method
      random_CRPS <- random_dataset$CRPS
      
      diff <- 100*(CRPS - random_CRPS) / random_CRPS
      
      # Append the Method and CRPS to the result_row
      result_CRPS <- cbind(result_CRPS, diff)
      result_CRPS_all <- cbind(result_CRPS_all, diff)
    }
  }
  
  # Calculate the mean of the CRPS for each method
  if (ncol(result_CRPS) > 2) 
  {
    result_CRPS_mean <- rowMeans(result_CRPS[, -1], na.rm=TRUE)
  }
  else
  {
    result_CRPS_mean <- result_CRPS$diff
  }
  
  # Append the result_CRPS_mean to result_row
  result_row[, methods] <- result_CRPS_mean
  
  # Append the result_row to the results_agg data frame
  results_agg <- rbind(results_agg, result_row)
}

# Reorder the columns in results_agg
results_agg <- results_agg[,c("task_id", "constant", "linear_regression", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]


# Print the new dataset with the Method and CRPS columns
print(results_agg)
results<-results_agg


# Change names
models <- methods
models_new_name <- c('const.', 'lin. reg.', 'RF', 'GBT', 'engression', 'MLP', 'ResNet', 'FT-Trans.', 'DRF', 'DGBT')
# Change names
colnames(results) <- c("task_id", models_new_name)

rowdiffs <- rowMeans(result_CRPS_all[,-1], na.rm=TRUE)
avg_diff <- data.frame(task_id="Avg. diff",const=rowdiffs[1], 'lin. reg.'=rowdiffs[9], RF=rowdiffs[8],GBT=rowdiffs[7], engression=rowdiffs[10], MLP=rowdiffs[2], ResNet=rowdiffs[3], FT=rowdiffs[4], DRF=rowdiffs[6], DGBT=rowdiffs[5])
colnames(avg_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_diff)

rowdiffs <- apply(result_CRPS_all[,-1], 1, median, na.rm=T)
avg_diff <- data.frame(task_id="Median diff",const=rowdiffs[1], 'lin. reg.'=rowdiffs[9], RF=rowdiffs[8],GBT=rowdiffs[7], engression=rowdiffs[10], MLP=rowdiffs[2], ResNet=rowdiffs[3], FT=rowdiffs[4], DRF=rowdiffs[6], DGBT=rowdiffs[5])
colnames(avg_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_diff)

results$"Avg diff"<-rowMeans(results[,-1], na.rm=TRUE)

## Create table
num_digits <- 3
output <- results
# Set the number of significant digits
output[, -1] <- signif(output[, -1], num_digits)

# Find the lowest value in each row
lowest_values <- apply(output[, -1], 1, function(x) min(x, na.rm=TRUE))

# Find the highest value in the second-to-last column
#highest_value <- max(output[nrow(output) - 1, -1], na.rm=TRUE)

# Convert numbers smaller than 0.1 and bigger than 100 to scientific notation
output[, -1][((abs(output[, -1]) < 0.1) | ((abs(output[, -1])) >=1000)) & 0==is.na(output[, -1])] <- format(output[, -1][((abs(output[, -1]) < 0.1) | ((abs(output[, -1])) >=1000)) & 0==is.na(output[, -1])], scientific = TRUE)
lowest_values[((abs(lowest_values) < 0.1) | ((abs(lowest_values)) >=1000)) & 0==is.na(lowest_values)] <- format(lowest_values[((abs(lowest_values) < 0.1) | ((abs(lowest_values)) >=1000)) & 0==is.na(lowest_values)], scientific=TRUE)

# Loop through each row and format the lowest value and highest value in bold
for (i in 1:nrow(output)) {
  output[i, -1] <- ifelse(output[i, -1] == lowest_values[i], paste0("\\textbf{", output[i, -1], "}"), output[i, -1])
}
#output[nrow(output) - 1, -1] <- ifelse(output[nrow(output) - 1, -1] == highest_value, paste0("\\textbf{", output[nrow(output) - 1, -1], "}"), output[nrow(output) - 1, -1])

caption <- paste0("Average relative difference of test CRPS using EP VS using IP, in \\%. 
                  Best results are bold. 
                  'Avg. diff.' denotes the average of each column.
                  'Median diff.' denotes the median of each column.")
#Bold results are non-inferior to the best result in a paired t-test at a 5\\% level.
label <- paste0("TABLES/table_results_CRPS_clustering_only_num_EP_VS_IP")
filename <- paste0(label, ".tex", sep="")
tab <- xtable(output, caption = caption, label = label)

print(tab, file = filename, sanitize.text.function = function(str) gsub("_", "\\_", str, fixed = TRUE),
      table.placement = getOption("xtable.table.placement", "ht!"), include.rownames = FALSE, hline.after = c(-1, -1, 0, (length(task_ids)), rep(dim(output)[1], 2)),
      size = "\\footnotesize", caption.placement = "bottom") #, floating.environment = "sidewaystable"




##############################################
########## ONLY UMAP ##################
##############################################
list_directories <- c("RESULTS/UMAP_DECOMPOSITION")
random_directory <- "RESULTS/RANDOM"
methods <- c('constant', 'MLP', 'ResNet', 'FTTrans', 'distributional_boosted_trees', 'drf', 'boosted_trees', 'rf', 'linear_regression', 'engression')

#### Define the unique task_ids
task_ids <- c()
for (directory in list_directories) {
  filenames <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
  for (filename in filenames) {
    task_id <- sub("^(\\d+)_.*", "\\1", basename(filename))
    task_ids <- c(task_ids, task_id)
  }
}
task_ids <- unique(task_ids)


# Create an empty data frame to store the results
results_agg <- data.frame(task_id = character(), stringsAsFactors = FALSE)

# Add a numeric column for each method
for (method in methods) {
  results_agg[[method]] <- numeric()
}

result_CRPS_all <- data.frame(method = methods, stringsAsFactors = FALSE)
for (task_id in task_ids)
{
  # Create a row for the current task_id
  result_row <- data.frame(task_id = task_id, stringsAsFactors = FALSE)
  
  # Add a column for each method
  for (method in methods) {
    result_row[[method]] <- NA
  }
  
  # Create a dataset for the CRPS with the correct number of rows
  result_CRPS <- data.frame(method = methods, stringsAsFactors = FALSE)
  
  # Iterate through the 4 directories
  for (directory in list_directories)
  {
    if (directory %in% c("RESULTS/CLUSTERING", "RESULTS/UMAP_DECOMPOSITION", "RESULTS/SPATIAL_DEPTH", "RESULTS/MAHALANOBIS"))
    {
      random_directory <- "RESULTS/RANDOM"
      i=1
    }
    else
    {
      random_directory <- "RESULTS/RANDOM_CAT"
      i=2
    }
    # Construct the filename for the current task_id and directory
    if (directory=="RESULTS/CLUSTERING")
    {
      filename <- file.path(directory, paste0(task_id, "_clustering_CRPS_results.csv"))
    }
    if (directory=="RESULTS/UMAP_DECOMPOSITION" | directory=="RESULTS/UMAP_DECOMPOSITION_CAT")
    {
      filename <- file.path(directory, paste0(task_id, "_umap_decomposition_CRPS_results.csv"))
    }
    if (directory=="RESULTS/SPATIAL_DEPTH")
    {
      filename <- file.path(directory, paste0(task_id, "_spatial_depth_CRPS_results.csv"))
    }
    if (directory=="RESULTS/MAHALANOBIS")
    {
      filename <- file.path(directory, paste0(task_id, "_mahalanobis_CRPS_results.csv"))
    }
    if (directory=="RESULTS/GOWER")
    {
      filename <- file.path(directory, paste0(task_id, "_gower_CRPS_results.csv"))
    }
    if (directory=="RESULTS/K_MEDOIDS")
    {
      filename <- file.path(directory, paste0(task_id, "_k_medoids_CRPS_results.csv"))
    }
    
    
    # Check if the file exists
    if (file.exists(filename))
    {
      # Load the results dataset
      results_dataset <- head(read.csv(filename),-2)
      
      # Extract the Method and CRPS columns
      method <- results_dataset$Method
      CRPS <- results_dataset$CRPS
      
      # Load the random file
      random_filename <- file.path(random_directory, paste0(task_id, "_random_crps_results.csv"))
      random_dataset <- head(read.csv(random_filename),-i)
      
      # Extract the Method and CRPS columns
      random_method <- random_dataset$Method
      random_CRPS <- random_dataset$CRPS
      
      diff <- 100*(CRPS - random_CRPS) / random_CRPS
      
      # Append the Method and CRPS to the result_row
      result_CRPS <- cbind(result_CRPS, diff)
      result_CRPS_all <- cbind(result_CRPS_all, diff)
    }
  }
  
  # Calculate the mean of the CRPS for each method
  if (ncol(result_CRPS) > 2) 
  {
    result_CRPS_mean <- rowMeans(result_CRPS[, -1], na.rm=TRUE)
  }
  else
  {
    result_CRPS_mean <- result_CRPS$diff
  }
  
  # Append the result_CRPS_mean to result_row
  result_row[, methods] <- result_CRPS_mean
  
  # Append the result_row to the results_agg data frame
  results_agg <- rbind(results_agg, result_row)
}

# Reorder the columns in results_agg
results_agg <- results_agg[,c("task_id", "constant", "linear_regression", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]


# Print the new dataset with the Method and CRPS columns
print(results_agg)
results<-results_agg


# Change names
models <- methods
models_new_name <- c('const.', 'lin. reg.', 'RF', 'GBT', 'engression', 'MLP', 'ResNet', 'FT-Trans.', 'DRF', 'DGBT')
# Change names
colnames(results) <- c("task_id", models_new_name)

rowdiffs <- rowMeans(result_CRPS_all[,-1], na.rm=TRUE)
avg_diff <- data.frame(task_id="Avg. diff",const=rowdiffs[1], 'lin. reg.'=rowdiffs[9], RF=rowdiffs[8],GBT=rowdiffs[7], engression=rowdiffs[10], MLP=rowdiffs[2], ResNet=rowdiffs[3], FT=rowdiffs[4], DRF=rowdiffs[6], DGBT=rowdiffs[5])
colnames(avg_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_diff)

rowdiffs <- apply(result_CRPS_all[,-1], 1, median, na.rm=T)
avg_diff <- data.frame(task_id="Median diff",const=rowdiffs[1], 'lin. reg.'=rowdiffs[9], RF=rowdiffs[8],GBT=rowdiffs[7], engression=rowdiffs[10], MLP=rowdiffs[2], ResNet=rowdiffs[3], FT=rowdiffs[4], DRF=rowdiffs[6], DGBT=rowdiffs[5])
colnames(avg_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_diff)

results$"Avg diff"<-rowMeans(results[,-1], na.rm=TRUE)

## Create table
num_digits <- 3
output <- results
# Set the number of significant digits
output[, -1] <- signif(output[, -1], num_digits)

# Find the lowest value in each row
lowest_values <- apply(output[, -1], 1, function(x) min(x, na.rm=TRUE))

# Find the highest value in the second-to-last column
#highest_value <- max(output[nrow(output) - 1, -1], na.rm=TRUE)

# Convert numbers smaller than 0.1 and bigger than 100 to scientific notation
output[, -1][((abs(output[, -1]) < 0.1) | ((abs(output[, -1])) >=1000)) & 0==is.na(output[, -1])] <- format(output[, -1][((abs(output[, -1]) < 0.1) | ((abs(output[, -1])) >=1000)) & 0==is.na(output[, -1])], scientific = TRUE)
lowest_values[((abs(lowest_values) < 0.1) | ((abs(lowest_values)) >=1000)) & 0==is.na(lowest_values)] <- format(lowest_values[((abs(lowest_values) < 0.1) | ((abs(lowest_values)) >=1000)) & 0==is.na(lowest_values)], scientific=TRUE)

# Loop through each row and format the lowest value and highest value in bold
for (i in 1:nrow(output)) {
  output[i, -1] <- ifelse(output[i, -1] == lowest_values[i], paste0("\\textbf{", output[i, -1], "}"), output[i, -1])
}
#output[nrow(output) - 1, -1] <- ifelse(output[nrow(output) - 1, -1] == highest_value, paste0("\\textbf{", output[nrow(output) - 1, -1], "}"), output[nrow(output) - 1, -1])

caption <- paste0("Average relative difference of test CRPS using EP VS using IP, in \\%. 
                  Best results are bold. 
                  'Avg. diff.' denotes the average of each column.
                  'Median diff.' denotes the median of each column.")
#Bold results are non-inferior to the best result in a paired t-test at a 5\\% level.
label <- paste0("TABLES/table_results_CRPS_umap_only_num_EP_VS_IP")
filename <- paste0(label, ".tex", sep="")
tab <- xtable(output, caption = caption, label = label)

print(tab, file = filename, sanitize.text.function = function(str) gsub("_", "\\_", str, fixed = TRUE),
      table.placement = getOption("xtable.table.placement", "ht!"), include.rownames = FALSE, hline.after = c(-1, -1, 0, (length(task_ids)), rep(dim(output)[1], 2)),
      size = "\\footnotesize", caption.placement = "bottom") #, floating.environment = "sidewaystable"





##############################################
########## ONLY SPATIAL DEPTH ###########
##############################################
list_directories <- c("RESULTS/SPATIAL_DEPTH")
random_directory <- "RESULTS/RANDOM"
methods <- c('constant', 'MLP', 'ResNet', 'FTTrans', 'distributional_boosted_trees', 'drf', 'boosted_trees', 'rf', 'linear_regression', 'engression')

#### Define the unique task_ids
task_ids <- c()
for (directory in list_directories) {
  filenames <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
  for (filename in filenames) {
    task_id <- sub("^(\\d+)_.*", "\\1", basename(filename))
    task_ids <- c(task_ids, task_id)
  }
}
task_ids <- unique(task_ids)


# Create an empty data frame to store the results
results_agg <- data.frame(task_id = character(), stringsAsFactors = FALSE)

# Add a numeric column for each method
for (method in methods) {
  results_agg[[method]] <- numeric()
}

result_CRPS_all <- data.frame(method = methods, stringsAsFactors = FALSE)
for (task_id in task_ids)
{
  # Create a row for the current task_id
  result_row <- data.frame(task_id = task_id, stringsAsFactors = FALSE)
  
  # Add a column for each method
  for (method in methods) {
    result_row[[method]] <- NA
  }
  
  # Create a dataset for the CRPS with the correct number of rows
  result_CRPS <- data.frame(method = methods, stringsAsFactors = FALSE)
  
  # Iterate through the 4 directories
  for (directory in list_directories)
  {
    if (directory %in% c("RESULTS/CLUSTERING", "RESULTS/UMAP_DECOMPOSITION", "RESULTS/SPATIAL_DEPTH", "RESULTS/MAHALANOBIS"))
    {
      random_directory <- "RESULTS/RANDOM"
      i=1
    }
    else
    {
      random_directory <- "RESULTS/RANDOM_CAT"
      i=2
    }
    # Construct the filename for the current task_id and directory
    if (directory=="RESULTS/CLUSTERING")
    {
      filename <- file.path(directory, paste0(task_id, "_clustering_CRPS_results.csv"))
    }
    if (directory=="RESULTS/UMAP_DECOMPOSITION" | directory=="RESULTS/UMAP_DECOMPOSITION_CAT")
    {
      filename <- file.path(directory, paste0(task_id, "_umap_decomposition_CRPS_results.csv"))
    }
    if (directory=="RESULTS/SPATIAL_DEPTH")
    {
      filename <- file.path(directory, paste0(task_id, "_spatial_depth_CRPS_results.csv"))
    }
    if (directory=="RESULTS/MAHALANOBIS")
    {
      filename <- file.path(directory, paste0(task_id, "_mahalanobis_CRPS_results.csv"))
    }
    if (directory=="RESULTS/GOWER")
    {
      filename <- file.path(directory, paste0(task_id, "_gower_CRPS_results.csv"))
    }
    if (directory=="RESULTS/K_MEDOIDS")
    {
      filename <- file.path(directory, paste0(task_id, "_k_medoids_CRPS_results.csv"))
    }
    
    
    # Check if the file exists
    if (file.exists(filename))
    {
      # Load the results dataset
      results_dataset <- head(read.csv(filename),-2)
      
      # Extract the Method and CRPS columns
      method <- results_dataset$Method
      CRPS <- results_dataset$CRPS
      
      # Load the random file
      random_filename <- file.path(random_directory, paste0(task_id, "_random_crps_results.csv"))
      random_dataset <- head(read.csv(random_filename),-i)
      
      # Extract the Method and CRPS columns
      random_method <- random_dataset$Method
      random_CRPS <- random_dataset$CRPS
      
      diff <- 100*(CRPS - random_CRPS) / random_CRPS
      
      # Append the Method and CRPS to the result_row
      result_CRPS <- cbind(result_CRPS, diff)
      result_CRPS_all <- cbind(result_CRPS_all, diff)
    }
  }
  
  # Calculate the mean of the CRPS for each method
  if (ncol(result_CRPS) > 2) 
  {
    result_CRPS_mean <- rowMeans(result_CRPS[, -1], na.rm=TRUE)
  }
  else
  {
    result_CRPS_mean <- result_CRPS$diff
  }
  
  # Append the result_CRPS_mean to result_row
  result_row[, methods] <- result_CRPS_mean
  
  # Append the result_row to the results_agg data frame
  results_agg <- rbind(results_agg, result_row)
}

# Reorder the columns in results_agg
results_agg <- results_agg[,c("task_id", "constant", "linear_regression", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]


# Print the new dataset with the Method and CRPS columns
print(results_agg)
results<-results_agg


# Change names
models <- methods
models_new_name <- c('const.', 'lin. reg.', 'RF', 'GBT', 'engression', 'MLP', 'ResNet', 'FT-Trans.', 'DRF', 'DGBT')
# Change names
colnames(results) <- c("task_id", models_new_name)

rowdiffs <- rowMeans(result_CRPS_all[,-1], na.rm=TRUE)
avg_diff <- data.frame(task_id="Avg. diff",const=rowdiffs[1], 'lin. reg.'=rowdiffs[9], RF=rowdiffs[8],GBT=rowdiffs[7], engression=rowdiffs[10], MLP=rowdiffs[2], ResNet=rowdiffs[3], FT=rowdiffs[4], DRF=rowdiffs[6], DGBT=rowdiffs[5])
colnames(avg_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_diff)

rowdiffs <- apply(result_CRPS_all[,-1], 1, median, na.rm=T)
avg_diff <- data.frame(task_id="Median diff",const=rowdiffs[1], 'lin. reg.'=rowdiffs[9], RF=rowdiffs[8],GBT=rowdiffs[7], engression=rowdiffs[10], MLP=rowdiffs[2], ResNet=rowdiffs[3], FT=rowdiffs[4], DRF=rowdiffs[6], DGBT=rowdiffs[5])
colnames(avg_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_diff)

results$"Avg diff"<-rowMeans(results[,-1], na.rm=TRUE)

## Create table
num_digits <- 3
output <- results
# Set the number of significant digits
output[, -1] <- signif(output[, -1], num_digits)

# Find the lowest value in each row
lowest_values <- apply(output[, -1], 1, function(x) min(x, na.rm=TRUE))

# Find the highest value in the second-to-last column
#highest_value <- max(output[nrow(output) - 1, -1], na.rm=TRUE)

# Convert numbers smaller than 0.1 and bigger than 100 to scientific notation
output[, -1][((abs(output[, -1]) < 0.1) | ((abs(output[, -1])) >=1000)) & 0==is.na(output[, -1])] <- format(output[, -1][((abs(output[, -1]) < 0.1) | ((abs(output[, -1])) >=1000)) & 0==is.na(output[, -1])], scientific = TRUE)
lowest_values[((abs(lowest_values) < 0.1) | ((abs(lowest_values)) >=1000)) & 0==is.na(lowest_values)] <- format(lowest_values[((abs(lowest_values) < 0.1) | ((abs(lowest_values)) >=1000)) & 0==is.na(lowest_values)], scientific=TRUE)

# Loop through each row and format the lowest value and highest value in bold
for (i in 1:nrow(output)) {
  output[i, -1] <- ifelse(output[i, -1] == lowest_values[i], paste0("\\textbf{", output[i, -1], "}"), output[i, -1])
}
#output[nrow(output) - 1, -1] <- ifelse(output[nrow(output) - 1, -1] == highest_value, paste0("\\textbf{", output[nrow(output) - 1, -1], "}"), output[nrow(output) - 1, -1])

caption <- paste0("Average relative difference of test CRPS using EP VS using IP, in \\%. 
                  Best results are bold. 
                  'Avg. diff.' denotes the average of each column.
                  'Median diff.' denotes the median of each column.")
#Bold results are non-inferior to the best result in a paired t-test at a 5\\% level.
label <- paste0("TABLES/table_results_CRPS_spatial_depth_only_num_EP_VS_IP")
filename <- paste0(label, ".tex", sep="")
tab <- xtable(output, caption = caption, label = label)

print(tab, file = filename, sanitize.text.function = function(str) gsub("_", "\\_", str, fixed = TRUE),
      table.placement = getOption("xtable.table.placement", "ht!"), include.rownames = FALSE, hline.after = c(-1, -1, 0, (length(task_ids)), rep(dim(output)[1], 2)),
      size = "\\footnotesize", caption.placement = "bottom") #, floating.environment = "sidewaystable"

###############################################
########## ONLY MAHALANOBIS ###################
###############################################
list_directories <- c("RESULTS/MAHALANOBIS")
random_directory <- "RESULTS/RANDOM"
methods <- c('constant', 'MLP', 'ResNet', 'FTTrans', 'distributional_boosted_trees', 'drf', 'boosted_trees', 'rf', 'linear_regression', 'engression')

#### Define the unique task_ids
task_ids <- c()
for (directory in list_directories) {
  filenames <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
  for (filename in filenames) {
    task_id <- sub("^(\\d+)_.*", "\\1", basename(filename))
    task_ids <- c(task_ids, task_id)
  }
}
task_ids <- unique(task_ids)


# Create an empty data frame to store the results
results_agg <- data.frame(task_id = character(), stringsAsFactors = FALSE)

# Add a numeric column for each method
for (method in methods) {
  results_agg[[method]] <- numeric()
}

result_CRPS_all <- data.frame(method = methods, stringsAsFactors = FALSE)
for (task_id in task_ids)
{
  # Create a row for the current task_id
  result_row <- data.frame(task_id = task_id, stringsAsFactors = FALSE)
  
  # Add a column for each method
  for (method in methods) {
    result_row[[method]] <- NA
  }
  
  # Create a dataset for the CRPS with the correct number of rows
  result_CRPS <- data.frame(method = methods, stringsAsFactors = FALSE)
  
  # Iterate through the 4 directories
  for (directory in list_directories)
  {
    if (directory %in% c("RESULTS/CLUSTERING", "RESULTS/UMAP_DECOMPOSITION", "RESULTS/SPATIAL_DEPTH", "RESULTS/MAHALANOBIS"))
    {
      random_directory <- "RESULTS/RANDOM"
      i=1
    }
    else
    {
      random_directory <- "RESULTS/RANDOM_CAT"
      i=2
    }
    # Construct the filename for the current task_id and directory
    if (directory=="RESULTS/CLUSTERING")
    {
      filename <- file.path(directory, paste0(task_id, "_clustering_CRPS_results.csv"))
    }
    if (directory=="RESULTS/UMAP_DECOMPOSITION" | directory=="RESULTS/UMAP_DECOMPOSITION_CAT")
    {
      filename <- file.path(directory, paste0(task_id, "_umap_decomposition_CRPS_results.csv"))
    }
    if (directory=="RESULTS/SPATIAL_DEPTH")
    {
      filename <- file.path(directory, paste0(task_id, "_spatial_depth_CRPS_results.csv"))
    }
    if (directory=="RESULTS/MAHALANOBIS")
    {
      filename <- file.path(directory, paste0(task_id, "_mahalanobis_CRPS_results.csv"))
    }
    if (directory=="RESULTS/GOWER")
    {
      filename <- file.path(directory, paste0(task_id, "_gower_CRPS_results.csv"))
    }
    if (directory=="RESULTS/K_MEDOIDS")
    {
      filename <- file.path(directory, paste0(task_id, "_k_medoids_CRPS_results.csv"))
    }
    
    
    # Check if the file exists
    if (file.exists(filename))
    {
      # Load the results dataset
      results_dataset <- head(read.csv(filename),-2)
      
      # Extract the Method and CRPS columns
      method <- results_dataset$Method
      CRPS <- results_dataset$CRPS
      
      # Load the random file
      random_filename <- file.path(random_directory, paste0(task_id, "_random_crps_results.csv"))
      random_dataset <- head(read.csv(random_filename),-i)
      
      # Extract the Method and CRPS columns
      random_method <- random_dataset$Method
      random_CRPS <- random_dataset$CRPS
      
      diff <- 100*(CRPS - random_CRPS) / random_CRPS
      
      # Append the Method and CRPS to the result_row
      result_CRPS <- cbind(result_CRPS, diff)
      result_CRPS_all <- cbind(result_CRPS_all, diff)
    }
  }
  
  # Calculate the mean of the CRPS for each method
  if (ncol(result_CRPS) > 2) 
  {
    result_CRPS_mean <- rowMeans(result_CRPS[, -1], na.rm=TRUE)
  }
  else
  {
    result_CRPS_mean <- result_CRPS$diff
  }
  
  # Append the result_CRPS_mean to result_row
  result_row[, methods] <- result_CRPS_mean
  
  # Append the result_row to the results_agg data frame
  results_agg <- rbind(results_agg, result_row)
}

# Reorder the columns in results_agg
results_agg <- results_agg[,c("task_id", "constant", "linear_regression", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]


# Print the new dataset with the Method and CRPS columns
print(results_agg)
results<-results_agg


# Change names
models <- methods
models_new_name <- c('const.', 'lin. reg.', 'RF', 'GBT', 'engression', 'MLP', 'ResNet', 'FT-Trans.', 'DRF', 'DGBT')
# Change names
colnames(results) <- c("task_id", models_new_name)

rowdiffs <- rowMeans(result_CRPS_all[,-1], na.rm=TRUE)
avg_diff <- data.frame(task_id="Avg. diff",const=rowdiffs[1], 'lin. reg.'=rowdiffs[9], RF=rowdiffs[8],GBT=rowdiffs[7], engression=rowdiffs[10], MLP=rowdiffs[2], ResNet=rowdiffs[3], FT=rowdiffs[4], DRF=rowdiffs[6], DGBT=rowdiffs[5])
colnames(avg_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_diff)

rowdiffs <- apply(result_CRPS_all[,-1], 1, median, na.rm=T)
avg_diff <- data.frame(task_id="Median diff",const=rowdiffs[1], 'lin. reg.'=rowdiffs[9], RF=rowdiffs[8],GBT=rowdiffs[7], engression=rowdiffs[10], MLP=rowdiffs[2], ResNet=rowdiffs[3], FT=rowdiffs[4], DRF=rowdiffs[6], DGBT=rowdiffs[5])
colnames(avg_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_diff)

results$"Avg diff"<-rowMeans(results[,-1], na.rm=TRUE)

## Create table
num_digits <- 3
output <- results
# Set the number of significant digits
output[, -1] <- signif(output[, -1], num_digits)

# Find the lowest value in each row
lowest_values <- apply(output[, -1], 1, function(x) min(x, na.rm=TRUE))

# Find the highest value in the second-to-last column
#highest_value <- max(output[nrow(output) - 1, -1], na.rm=TRUE)

# Convert numbers smaller than 0.1 and bigger than 100 to scientific notation
output[, -1][((abs(output[, -1]) < 0.1) | ((abs(output[, -1])) >=1000)) & 0==is.na(output[, -1])] <- format(output[, -1][((abs(output[, -1]) < 0.1) | ((abs(output[, -1])) >=1000)) & 0==is.na(output[, -1])], scientific = TRUE)
lowest_values[((abs(lowest_values) < 0.1) | ((abs(lowest_values)) >=1000)) & 0==is.na(lowest_values)] <- format(lowest_values[((abs(lowest_values) < 0.1) | ((abs(lowest_values)) >=1000)) & 0==is.na(lowest_values)], scientific=TRUE)

# Loop through each row and format the lowest value and highest value in bold
for (i in 1:nrow(output)) {
  output[i, -1] <- ifelse(output[i, -1] == lowest_values[i], paste0("\\textbf{", output[i, -1], "}"), output[i, -1])
}
#output[nrow(output) - 1, -1] <- ifelse(output[nrow(output) - 1, -1] == highest_value, paste0("\\textbf{", output[nrow(output) - 1, -1], "}"), output[nrow(output) - 1, -1])

caption <- paste0("Average relative difference of test CRPS using EP VS using IP, in \\%. 
                  Best results are bold. 
                  'Avg. diff.' denotes the average of each column.
                  'Median diff.' denotes the median of each column.")
#Bold results are non-inferior to the best result in a paired t-test at a 5\\% level.
label <- paste0("TABLES/table_results_CRPS_mahalanobis_only_num_EP_VS_IP")
filename <- paste0(label, ".tex", sep="")
tab <- xtable(output, caption = caption, label = label)

print(tab, file = filename, sanitize.text.function = function(str) gsub("_", "\\_", str, fixed = TRUE),
      table.placement = getOption("xtable.table.placement", "ht!"), include.rownames = FALSE, hline.after = c(-1, -1, 0, (length(task_ids)), rep(dim(output)[1], 2)),
      size = "\\footnotesize", caption.placement = "bottom") #, floating.environment = "sidewaystable"
