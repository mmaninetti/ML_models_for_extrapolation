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

df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

list_directories <- c("RESULTS/K_MEDOIDS", "RESULTS/UMAP_DECOMPOSITION", "RESULTS/GOWER")
methods <- c('constant', 'MLP', 'ResNet', 'FTTrans', 'distributional_boosted_trees', 'drf', 'boosted_trees', 'rf', 'linear_regression', 'engression', 'GAM')

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
    # Construct the filename for the current task_id and directory
    if (directory=="RESULTS/K_MEDOIDS")
    {
      filename <- file.path(directory, paste0(task_id, "_k_medoids_CRPS_results.csv"))
    }
    if (directory=="RESULTS/UMAP_DECOMPOSITION")
    {
      filename <- file.path(directory, paste0(task_id, "_umap_decomposition_CRPS_results.csv"))
    }
    if (directory=="RESULTS/GOWER")
    {
      filename <- file.path(directory, paste0(task_id, "_gower_CRPS_results.csv"))
    }
    
    # Check if the file exists
    if (file.exists(filename))
    {
      # Load the results dataset
      results_dataset <- head(read.csv(filename),-1)
      
      # Extract the Method and CRPS columns
      method <- results_dataset$Method
      CRPS <- results_dataset$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      

      # Append the Method and CRPS to the result_row
      result_CRPS <- cbind(result_CRPS, CRPS)
    }
  }
  
  # Calculate the mean of the CRPS for each method
  if (ncol(result_CRPS) > 2) 
  {
    result_CRPS_mean <- rowMeans(result_CRPS[, -1], na.rm=TRUE)
  }
  else
  {
    result_CRPS_mean <- result_CRPS$CRPS
  }

  # Append the result_CRPS_mean to result_row
  result_row[, methods] <- result_CRPS_mean
  
  # Append the result_row to the results_agg data frame
  results_agg <- rbind(results_agg, result_row)
}

# Reorder the columns in results_agg
results_agg <- results_agg[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]


# Print the new dataset with the Method and CRPS columns
print(results_agg)
results<-results_agg


# Change names
models <- methods
models_new_name <- c('const.', 'lin. reg.', 'GAM', 'RF', 'GBT', 'engression', 'MLP', 'ResNet', 'FT-Trans.', 'DRF', 'DGBT')
# Change names
colnames(results) <- c("task_id", models_new_name)


#### Compute relative differences
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

# Loop through each directory
for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the lowest CRPS
      lowest_CRPS <- min(CRPS, na.rm=TRUE)
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,]<-(CRPS - lowest_CRPS) / lowest_CRPS
      df <- rbind(df, tmp)
    }
  }
}

# Set the row names of the data frame as the Method column
mean_rel_diff <- 100*colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_rel_diff <- data.frame(task_id = "Avg. diff.", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_rel_diff[[method]] <- mean_rel_diff[i]
 i=i+1
}
avg_rel_diff <- avg_rel_diff[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_rel_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_rel_diff)

#### Compute normalized accuracy
# Loop through each directory
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the lowest CRPS
      mid_CRPS <- sort(CRPS, decreasing = TRUE, na.last=NA)[3]
      lowest_CRPS <- min(CRPS, na.rm=TRUE)
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,] <- pmin(pmax((mid_CRPS - CRPS) / (mid_CRPS - lowest_CRPS), 0), 1)
      df <- rbind(df, tmp)
    }
  }
}

# Set the row names of the data frame as the Method column
mean_norm_acc <- 100*colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_norm_acc <- data.frame(task_id = "Avg. acc.", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_norm_acc[[method]] <- mean_norm_acc[i]
 i=i+1
}
avg_norm_acc <- avg_norm_acc[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_norm_acc) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_norm_acc)

#### Compute ranks
# Loop through each directory
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,]<-rank(CRPS, na.last="keep")
      df <- rbind(df, tmp)
    }
  }
}
# Set the row names of the data frame as the Method column
mean_rank <- colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_rank <- data.frame(task_id = "Avg. rank", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_rank[[method]] <- mean_rank[i]
  i=i+1
}
avg_rank <- avg_rank[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_rank) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_rank)

## Create table
num_digits <- 3
output <- results
# Set the number of significant digits
output[, -1] <- signif(output[, -1], num_digits)

# Find the lowest value in each row
lowest_values <- apply(output[, -1], 1, function(x) min(x, na.rm=TRUE))

# Find the highest value in the second-to-last column
highest_value <- max(output[nrow(output) - 1, -1], na.rm=TRUE)

# Convert numbers smaller than 0.1 and bigger than 100 to scientific notation
output[, -1][(output[, -1] < 0.1 | output[, -1] >=1000) & 0==is.na(output[, -1])] <- format(output[, -1][(output[, -1] < 0.1 | output[, -1] >=1000) & 0==is.na(output[, -1])], scientific = TRUE)
lowest_values[(lowest_values<0.1 | lowest_values>=1000) & 0==is.na(lowest_values)] <- format(lowest_values[(lowest_values<0.1 | lowest_values>=1000) & 0==is.na(lowest_values)], scientific=TRUE)

# Loop through each row and format the lowest value and highest value in bold
for (i in 1:nrow(output)) {
  if (i!=nrow(output) - 1)
  {
  output[i, -1] <- ifelse(output[i, -1] == lowest_values[i], paste0("\\textbf{", output[i, -1], "}"), output[i, -1])
  }
}
output[nrow(output) - 1, -1] <- ifelse(output[nrow(output) - 1, -1] == highest_value, paste0("\\textbf{", output[nrow(output) - 1, -1], "}"), output[nrow(output) - 1, -1])

caption <- paste0("Average test CRPS. 
                  Best results are bold. 
                  'Avg. diff.' denotes the average relative difference in \\% of a method compared to the best method.
                  'Avg. acc.' denotes the average normalized accuracy in \\% of a method.
                  'Avg. rank' denotes the average rank of a method.")
                   #Bold results are non-inferior to the best result in a paired t-test at a 5\\% level.
label <- paste0("TABLES/table_results_CRPS")
filename <- paste0(label, ".tex", sep="")
tab <- xtable(output, caption = caption, label = label)

print(tab, file = filename, sanitize.text.function = function(str) gsub("_", "\\_", str, fixed = TRUE),
      table.placement = getOption("xtable.table.placement", "ht!"), include.rownames = FALSE, hline.after = c(-1, -1, 0, (length(task_ids)), rep(dim(output)[1], 2)),
      size = "\\footnotesize", caption.placement = "bottom") #, floating.environment = "sidewaystable"



###############################################
############ ONLY K_MEDOIDS ###################
###############################################
list_directories <- c("RESULTS/K_MEDOIDS")
methods <- c('constant', 'MLP', 'ResNet', 'FTTrans', 'distributional_boosted_trees', 'drf', 'boosted_trees', 'rf', 'linear_regression', 'engression', 'GAM')

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
    # Construct the filename for the current task_id and directory
    if (directory=="RESULTS/K_MEDOIDS")
    {
      filename <- file.path(directory, paste0(task_id, "_k_medoids_CRPS_results.csv"))
    }
    if (directory=="RESULTS/UMAP_DECOMPOSITION")
    {
      filename <- file.path(directory, paste0(task_id, "_umap_decomposition_CRPS_results.csv"))
    }
    if (directory=="RESULTS/GOWER")
    {
      filename <- file.path(directory, paste0(task_id, "_gower_CRPS_results.csv"))
    }
    
    # Check if the file exists
    if (file.exists(filename))
    {
      # Load the results dataset
      results_dataset <- head(read.csv(filename),-1)
      
      # Extract the Method and CRPS columns
      method <- results_dataset$Method
      CRPS <- results_dataset$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      

      # Append the Method and CRPS to the result_row
      result_CRPS <- cbind(result_CRPS, CRPS)
    }
  }
  
  # Calculate the mean of the CRPS for each method
  if (ncol(result_CRPS) > 2) 
  {
    result_CRPS_mean <- rowMeans(result_CRPS[, -1], na.rm=TRUE)
  }
  else
  {
    result_CRPS_mean <- result_CRPS$CRPS
  }

  # Append the result_CRPS_mean to result_row
  result_row[, methods] <- result_CRPS_mean
  
  # Append the result_row to the results_agg data frame
  results_agg <- rbind(results_agg, result_row)
}

# Reorder the columns in results_agg
results_agg <- results_agg[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]


# Print the new dataset with the Method and CRPS columns
print(results_agg)
results<-results_agg


# Change names
models <- methods
models_new_name <- c('const.', 'lin. reg.', 'GAM', 'RF', 'GBT', 'engression', 'MLP', 'ResNet', 'FT-Trans.', 'DRF', 'DGBT')
# Change names
colnames(results) <- c("task_id", models_new_name)


#### Compute relative differences
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

# Loop through each directory
for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the lowest CRPS
      lowest_CRPS <- min(CRPS, na.rm=TRUE)
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,]<-(CRPS - lowest_CRPS) / lowest_CRPS
      df <- rbind(df, tmp)
    }
  }
}

# Set the row names of the data frame as the Method column
mean_rel_diff <- 100*colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_rel_diff <- data.frame(task_id = "Avg. diff.", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_rel_diff[[method]] <- mean_rel_diff[i]
 i=i+1
}
avg_rel_diff <- avg_rel_diff[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_rel_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_rel_diff)

#### Compute normalized accuracy
# Loop through each directory
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the lowest CRPS
      mid_CRPS <- sort(CRPS, decreasing = TRUE, na.last=NA)[3]
      lowest_CRPS <- min(CRPS, na.rm=TRUE)
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,] <- pmin(pmax((mid_CRPS - CRPS) / (mid_CRPS - lowest_CRPS), 0), 1)
      df <- rbind(df, tmp)
    }
  }
}

# Set the row names of the data frame as the Method column
mean_norm_acc <- 100*colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_norm_acc <- data.frame(task_id = "Avg. acc.", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_norm_acc[[method]] <- mean_norm_acc[i]
 i=i+1
}
avg_norm_acc <- avg_norm_acc[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_norm_acc) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_norm_acc)

#### Compute ranks
# Loop through each directory
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,]<-rank(CRPS, na.last="keep")
      df <- rbind(df, tmp)
    }
  }
}
# Set the row names of the data frame as the Method column
mean_rank <- colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_rank <- data.frame(task_id = "Avg. rank", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_rank[[method]] <- mean_rank[i]
  i=i+1
}
avg_rank <- avg_rank[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_rank) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_rank)

## Create table
num_digits <- 3
output <- results
# Set the number of significant digits
output[, -1] <- signif(output[, -1], num_digits)

# Find the lowest value in each row
lowest_values <- apply(output[, -1], 1, function(x) min(x, na.rm=TRUE))

# Find the highest value in the second-to-last column
highest_value <- max(output[nrow(output) - 1, -1], na.rm=TRUE)

# Convert numbers smaller than 0.1 and bigger than 100 to scientific notation
output[, -1][(output[, -1] < 0.1 | output[, -1] >=1000) & 0==is.na(output[, -1])] <- format(output[, -1][(output[, -1] < 0.1 | output[, -1] >=1000) & 0==is.na(output[, -1])], scientific = TRUE)
lowest_values[(lowest_values<0.1 | lowest_values>=1000) & 0==is.na(lowest_values)] <- format(lowest_values[(lowest_values<0.1 | lowest_values>=1000) & 0==is.na(lowest_values)], scientific=TRUE)

# Loop through each row and format the lowest value and highest value in bold
for (i in 1:nrow(output)) {
  if (i!=nrow(output) - 1)
  {
  output[i, -1] <- ifelse(output[i, -1] == lowest_values[i], paste0("\\textbf{", output[i, -1], "}"), output[i, -1])
  }
}
output[nrow(output) - 1, -1] <- ifelse(output[nrow(output) - 1, -1] == highest_value, paste0("\\textbf{", output[nrow(output) - 1, -1], "}"), output[nrow(output) - 1, -1])

caption <- paste0("Average test CRPS. 
                  Best results are bold. 
                  'Avg. diff.' denotes the average relative difference in \\% of a method compared to the best method.
                  'Avg. acc.' denotes the average normalized accuracy in \\% of a method.
                  'Avg. rank' denotes the average rank of a method.")
                   #Bold results are non-inferior to the best result in a paired t-test at a 5\\% level.
label <- paste0("TABLES/table_results_CRPS_k_medoids")
filename <- paste0(label, ".tex", sep="")
tab <- xtable(output, caption = caption, label = label)

print(tab, file = filename, sanitize.text.function = function(str) gsub("_", "\\_", str, fixed = TRUE),
      table.placement = getOption("xtable.table.placement", "ht!"), include.rownames = FALSE, hline.after = c(-1, -1, 0, (length(task_ids)), rep(dim(output)[1], 2)),
      size = "\\footnotesize", caption.placement = "bottom") #, floating.environment = "sidewaystable"


###############################################
############ ONLY GOWER #######################
###############################################
list_directories <- c("RESULTS/GOWER")
methods <- c('constant', 'MLP', 'ResNet', 'FTTrans', 'distributional_boosted_trees', 'drf', 'boosted_trees', 'rf', 'linear_regression', 'engression', 'GAM')

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
    # Construct the filename for the current task_id and directory
    if (directory=="RESULTS/K_MEDOIDS")
    {
      filename <- file.path(directory, paste0(task_id, "_k_medoids_CRPS_results.csv"))
    }
    if (directory=="RESULTS/UMAP_DECOMPOSITION")
    {
      filename <- file.path(directory, paste0(task_id, "_umap_decomposition_CRPS_results.csv"))
    }
    if (directory=="RESULTS/GOWER")
    {
      filename <- file.path(directory, paste0(task_id, "_gower_CRPS_results.csv"))
    }
    
    # Check if the file exists
    if (file.exists(filename))
    {
      # Load the results dataset
      results_dataset <- head(read.csv(filename),-1)
      
      # Extract the Method and CRPS columns
      method <- results_dataset$Method
      CRPS <- results_dataset$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      

      # Append the Method and CRPS to the result_row
      result_CRPS <- cbind(result_CRPS, CRPS)
    }
  }
  
  # Calculate the mean of the CRPS for each method
  if (ncol(result_CRPS) > 2) 
  {
    result_CRPS_mean <- rowMeans(result_CRPS[, -1], na.rm=TRUE)
  }
  else
  {
    result_CRPS_mean <- result_CRPS$CRPS
  }

  # Append the result_CRPS_mean to result_row
  result_row[, methods] <- result_CRPS_mean
  
  # Append the result_row to the results_agg data frame
  results_agg <- rbind(results_agg, result_row)
}

# Reorder the columns in results_agg
results_agg <- results_agg[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]


# Print the new dataset with the Method and CRPS columns
print(results_agg)
results<-results_agg


# Change names
models <- methods
models_new_name <- c('const.', 'lin. reg.', 'GAM', 'RF', 'GBT', 'engression', 'MLP', 'ResNet', 'FT-Trans.', 'DRF', 'DGBT')
# Change names
colnames(results) <- c("task_id", models_new_name)


#### Compute relative differences
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

# Loop through each directory
for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the lowest CRPS
      lowest_CRPS <- min(CRPS, na.rm=TRUE)
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,]<-(CRPS - lowest_CRPS) / lowest_CRPS
      df <- rbind(df, tmp)
    }
  }
}

# Set the row names of the data frame as the Method column
mean_rel_diff <- 100*colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_rel_diff <- data.frame(task_id = "Avg. diff.", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_rel_diff[[method]] <- mean_rel_diff[i]
 i=i+1
}
avg_rel_diff <- avg_rel_diff[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_rel_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_rel_diff)

#### Compute normalized accuracy
# Loop through each directory
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the lowest CRPS
      mid_CRPS <- sort(CRPS, decreasing = TRUE, na.last=NA)[3]
      lowest_CRPS <- min(CRPS, na.rm=TRUE)
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,] <- pmin(pmax((mid_CRPS - CRPS) / (mid_CRPS - lowest_CRPS), 0), 1)
      df <- rbind(df, tmp)
    }
  }
}

# Set the row names of the data frame as the Method column
mean_norm_acc <- 100*colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_norm_acc <- data.frame(task_id = "Avg. acc.", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_norm_acc[[method]] <- mean_norm_acc[i]
 i=i+1
}
avg_norm_acc <- avg_norm_acc[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_norm_acc) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_norm_acc)

#### Compute ranks
# Loop through each directory
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,]<-rank(CRPS, na.last="keep")
      df <- rbind(df, tmp)
    }
  }
}
# Set the row names of the data frame as the Method column
mean_rank <- colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_rank <- data.frame(task_id = "Avg. rank", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_rank[[method]] <- mean_rank[i]
  i=i+1
}
avg_rank <- avg_rank[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_rank) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_rank)

## Create table
num_digits <- 3
output <- results
# Set the number of significant digits
output[, -1] <- signif(output[, -1], num_digits)

# Find the lowest value in each row
lowest_values <- apply(output[, -1], 1, function(x) min(x, na.rm=TRUE))

# Find the highest value in the second-to-last column
highest_value <- max(output[nrow(output) - 1, -1], na.rm=TRUE)

# Convert numbers smaller than 0.1 and bigger than 100 to scientific notation
output[, -1][(output[, -1] < 0.1 | output[, -1] >=1000) & 0==is.na(output[, -1])] <- format(output[, -1][(output[, -1] < 0.1 | output[, -1] >=1000) & 0==is.na(output[, -1])], scientific = TRUE)
lowest_values[(lowest_values<0.1 | lowest_values>=1000) & 0==is.na(lowest_values)] <- format(lowest_values[(lowest_values<0.1 | lowest_values>=1000) & 0==is.na(lowest_values)], scientific=TRUE)

# Loop through each row and format the lowest value and highest value in bold
for (i in 1:nrow(output)) {
  if (i!=nrow(output) - 1)
  {
  output[i, -1] <- ifelse(output[i, -1] == lowest_values[i], paste0("\\textbf{", output[i, -1], "}"), output[i, -1])
  }
}
output[nrow(output) - 1, -1] <- ifelse(output[nrow(output) - 1, -1] == highest_value, paste0("\\textbf{", output[nrow(output) - 1, -1], "}"), output[nrow(output) - 1, -1])

caption <- paste0("Average test CRPS. 
                  Best results are bold. 
                  'Avg. diff.' denotes the average relative difference in \\% of a method compared to the best method.
                  'Avg. acc.' denotes the average normalized accuracy in \\% of a method.
                  'Avg. rank' denotes the average rank of a method.")
                   #Bold results are non-inferior to the best result in a paired t-test at a 5\\% level.
label <- paste0("TABLES/table_results_CRPS_gower")
filename <- paste0(label, ".tex", sep="")
tab <- xtable(output, caption = caption, label = label)

print(tab, file = filename, sanitize.text.function = function(str) gsub("_", "\\_", str, fixed = TRUE),
      table.placement = getOption("xtable.table.placement", "ht!"), include.rownames = FALSE, hline.after = c(-1, -1, 0, (length(task_ids)), rep(dim(output)[1], 2)),
      size = "\\footnotesize", caption.placement = "bottom") #, floating.environment = "sidewaystable"



###############################################
############ ONLY UMAP ########################
###############################################
list_directories <- c("RESULTS/UMAP_DECOMPOSITION")
methods <- c('constant', 'MLP', 'ResNet', 'FTTrans', 'distributional_boosted_trees', 'drf', 'boosted_trees', 'rf', 'linear_regression', 'engression', 'GAM')

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
    # Construct the filename for the current task_id and directory
    if (directory=="RESULTS/K_MEDOIDS")
    {
      filename <- file.path(directory, paste0(task_id, "_k_medoids_CRPS_results.csv"))
    }
    if (directory=="RESULTS/UMAP_DECOMPOSITION")
    {
      filename <- file.path(directory, paste0(task_id, "_umap_decomposition_CRPS_results.csv"))
    }
    if (directory=="RESULTS/GOWER")
    {
      filename <- file.path(directory, paste0(task_id, "_gower_CRPS_results.csv"))
    }
    
    # Check if the file exists
    if (file.exists(filename))
    {
      # Load the results dataset
      results_dataset <- head(read.csv(filename),-1)
      
      # Extract the Method and CRPS columns
      method <- results_dataset$Method
      CRPS <- results_dataset$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      

      # Append the Method and CRPS to the result_row
      result_CRPS <- cbind(result_CRPS, CRPS)
    }
  }
  
  # Calculate the mean of the CRPS for each method
  if (ncol(result_CRPS) > 2) 
  {
    result_CRPS_mean <- rowMeans(result_CRPS[, -1], na.rm=TRUE)
  }
  else
  {
    result_CRPS_mean <- result_CRPS$CRPS
  }

  # Append the result_CRPS_mean to result_row
  result_row[, methods] <- result_CRPS_mean
  
  # Append the result_row to the results_agg data frame
  results_agg <- rbind(results_agg, result_row)
}

# Reorder the columns in results_agg
results_agg <- results_agg[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]


# Print the new dataset with the Method and CRPS columns
print(results_agg)
results<-results_agg


# Change names
models <- methods
models_new_name <- c('const.', 'lin. reg.', 'GAM', 'RF', 'GBT', 'engression', 'MLP', 'ResNet', 'FT-Trans.', 'DRF', 'DGBT')
# Change names
colnames(results) <- c("task_id", models_new_name)


#### Compute relative differences
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

# Loop through each directory
for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the lowest CRPS
      lowest_CRPS <- min(CRPS, na.rm=TRUE)
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,]<-(CRPS - lowest_CRPS) / lowest_CRPS
      df <- rbind(df, tmp)
    }
  }
}

# Set the row names of the data frame as the Method column
mean_rel_diff <- 100*colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_rel_diff <- data.frame(task_id = "Avg. diff.", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_rel_diff[[method]] <- mean_rel_diff[i]
 i=i+1
}
avg_rel_diff <- avg_rel_diff[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_rel_diff) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_rel_diff)

#### Compute normalized accuracy
# Loop through each directory
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the lowest CRPS
      mid_CRPS <- sort(CRPS, decreasing = TRUE, na.last=NA)[3]
      lowest_CRPS <- min(CRPS, na.rm=TRUE)
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,] <- pmin(pmax((mid_CRPS - CRPS) / (mid_CRPS - lowest_CRPS), 0), 1)
      df <- rbind(df, tmp)
    }
  }
}

# Set the row names of the data frame as the Method column
mean_norm_acc <- 100*colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_norm_acc <- data.frame(task_id = "Avg. acc.", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_norm_acc[[method]] <- mean_norm_acc[i]
 i=i+1
}
avg_norm_acc <- avg_norm_acc[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_norm_acc) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_norm_acc)

#### Compute ranks
# Loop through each directory
df <- data.frame()
for (method in methods) {
  df[[method]] <- numeric()
}

for (directory in list_directories) {
  # Loop through each file in the directory
  for (filename in list.files(directory)) {
    # Check if the file is a CSV file
    if (endsWith(filename, ".csv")) {
      # Extract the task name and file path
      task_name <- filename
      filepath <- file.path(directory, filename)
      
      # Read the CSV file into a data frame
      table <- head(read.csv(filepath),-1)
      
      # Extract the CRPS column
      CRPS <- table$CRPS

      CRPS <- ifelse(CRPS >= 0, CRPS, NA)
      
      
      
      # Calculate the normalized CRPS and add it to the data frame
      tmp <- data.frame()
      for (method in methods) {
        tmp[[method]] <- numeric()
      }
      tmp[1,]<-rank(CRPS, na.last="keep")
      df <- rbind(df, tmp)
    }
  }
}
# Set the row names of the data frame as the Method column
mean_rank <- colMeans(df, na.rm=TRUE)
# Create a row for the current task_id
avg_rank <- data.frame(task_id = "Avg. rank", stringsAsFactors = FALSE)
# Add a column for each method
i=1
for (method in methods) {
  avg_rank[[method]] <- mean_rank[i]
  i=i+1
}
avg_rank <- avg_rank[,c("task_id", "constant", "linear_regression", "GAM", "rf", "boosted_trees", "engression", "MLP", "ResNet", "FTTrans", "drf", "distributional_boosted_trees")]
colnames(avg_rank) <- c("task_id", models_new_name)
results <- bind_rows(results, avg_rank)

## Create table
num_digits <- 3
output <- results
# Set the number of significant digits
output[, -1] <- signif(output[, -1], num_digits)

# Find the lowest value in each row
lowest_values <- apply(output[, -1], 1, function(x) min(x, na.rm=TRUE))

# Find the highest value in the second-to-last column
highest_value <- max(output[nrow(output) - 1, -1], na.rm=TRUE)

# Convert numbers smaller than 0.1 and bigger than 100 to scientific notation
output[, -1][(output[, -1] < 0.1 | output[, -1] >=1000) & 0==is.na(output[, -1])] <- format(output[, -1][(output[, -1] < 0.1 | output[, -1] >=1000) & 0==is.na(output[, -1])], scientific = TRUE)
lowest_values[(lowest_values<0.1 | lowest_values>=1000) & 0==is.na(lowest_values)] <- format(lowest_values[(lowest_values<0.1 | lowest_values>=1000) & 0==is.na(lowest_values)], scientific=TRUE)

# Loop through each row and format the lowest value and highest value in bold
for (i in 1:nrow(output)) {
  if (i!=nrow(output) - 1)
  {
  output[i, -1] <- ifelse(output[i, -1] == lowest_values[i], paste0("\\textbf{", output[i, -1], "}"), output[i, -1])
  }
}
output[nrow(output) - 1, -1] <- ifelse(output[nrow(output) - 1, -1] == highest_value, paste0("\\textbf{", output[nrow(output) - 1, -1], "}"), output[nrow(output) - 1, -1])

caption <- paste0("Average test CRPS. 
                  Best results are bold. 
                  'Avg. diff.' denotes the average relative difference in \\% of a method compared to the best method.
                  'Avg. acc.' denotes the average normalized accuracy in \\% of a method.
                  'Avg. rank' denotes the average rank of a method.")
                   #Bold results are non-inferior to the best result in a paired t-test at a 5\\% level.
label <- paste0("TABLES/table_results_CRPS_umap")
filename <- paste0(label, ".tex", sep="")
tab <- xtable(output, caption = caption, label = label)

print(tab, file = filename, sanitize.text.function = function(str) gsub("_", "\\_", str, fixed = TRUE),
      table.placement = getOption("xtable.table.placement", "ht!"), include.rownames = FALSE, hline.after = c(-1, -1, 0, (length(task_ids)), rep(dim(output)[1], 2)),
      size = "\\footnotesize", caption.placement = "bottom") #, floating.environment = "sidewaystable"

