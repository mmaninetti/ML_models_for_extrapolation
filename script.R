
if (!requireNamespace("ddalpha", quietly = TRUE)) {
  install.packages("ddalpha")
}
library(ddalpha)

# Load your DataFrame from the Python environment
my_data <- my_data  # Make sure to use the same name you assigned in Python

# Compute the simplicial depth with respect to the centroid
simplicial_depth <- depth.simplicial(my_data)

# Print the result
print(simplicial_depth)
