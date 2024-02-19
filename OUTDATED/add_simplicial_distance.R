library(ddalpha)
tasks<-c(361072, 361073, 361074, 361076, 361077, 361078, 361079, 361080, 361081, 361082, 361083, 361084, 361085, 361086, 361087, 361088, 361279, 361280, 361281)
for (task in tasks){
  print(task)
  data<-read.csv(paste0("C:/Users/dalma/Desktop/THESIS_ETH/DATASETS/Regression_on_numerical_features/",as.character(task)))
  simplicial_depth <- depth.simplicial(as.matrix(data), as.matrix(data))
}


impute.mean <- function(x) replace(x, is.na(x) | is.nan(x) | is.infinite(x), mean(x[!is.na(x) & !is.nan(x) & !is.infinite(x)]))
data2 <- apply(as.matrix(data), 2, impute.mean)