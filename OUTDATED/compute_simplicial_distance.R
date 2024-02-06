#! /usr/bin/Rscript
if (!requireNamespace("ddalpha", quietly = TRUE)) {
  install.packages("ddalpha")
}
library(ddalpha)

# Load your DataFrame from the Python environment
r_dataframe <- r_dataframe  # Make sure to use the same name you assigned in Python

# Compute the simplicial depth with respect to the centroid
simplicial_depth <- depth.simplicial(r_dataframe, r_dataframe)