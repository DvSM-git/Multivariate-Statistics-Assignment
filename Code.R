rm(list=ls())

library(robustbase)
library(parallel)

# Simulation parameters
set.seed(0)
n_train <- 100
n_test <- 1000 
n_sim <- 1000
beta <- c(0, 1, 1, 1)
p <- length(beta) - 1

# --- Single simulation step function --- #
sim_step <- function(i, n_train, n_test, p, beta) {
  require(robustbase)
  
  # ---Data Generation (Common Structures) --- #
  
  # Generate Design Matrices once
  X_train <- cbind(1, matrix(rnorm(n_train * p), ncol = p))
  X_test  <- cbind(1, matrix(rnorm(n_test * p), ncol = p))
  
  # True Structural Part
  y_true_train <- X_train %*% beta
  y_test <- X_test %*% beta
  
  # ---  Helper Function for Estimation --- #
  
  # Reduces code duplication for OLS vs Robust steps
  # TODO Add more estimators
  evaluate_models <- function(y_train, X_train) {
    # OLS
    b_ols <- .lm.fit(X_train, y_train)$coefficients
    # M-estimator
    b_M <- coef(lmrob(y_train ~ 0 + X_train))
    # MM-estimator with huber psi
    b_MM <- coef(MASS::rlm(y_train ~ 0 + X_train), method = "M", psi = psi.huber)
    # Trimmed least squares estimator
    b_tls <- coef(ltsReg(y_train ~ X_train[, -1, drop = FALSE], nsamp = "best"))
    
    # Prediction & RMSE
    rmse_ols <- sqrt(mean((y_test - (X_test %*% b_ols))^2))
    rmse_M <- sqrt(mean((y_test - (X_test %*% b_M))^2))
    rmse_MM <- sqrt(mean((y_test - (X_test %*% b_MM))^2))
    rmse_tls <- sqrt(mean((y_test - (X_test %*% b_tls))^2))
    
    return(c(rmse_ols, rmse_M, rmse_MM, rmse_tls))
  }
  
  # --- Normal Errors --- #
  y_normal <- y_true_train + rnorm(n_train)
  res_normal <- evaluate_models(y_normal, X_train)
  
  # --- Tukey-Huber Contaminated Errors --- #
  # Parameters
  alpha <- 0.1
  m <- 0
  k <- 10
  
  # Model
  is_outlier <- rbinom(n_train, 1, alpha)
  err_tukey <- rnorm(n_train) * (1 - is_outlier) + rnorm(n_train, m, k) * is_outlier
  y_tukey <- y_true_train + err_tukey
  
  res_tukey <- evaluate_models(y_tukey, X_train)
  
  # --- t(1) distributed errors --- #
  y_t1 <- y_true_train + rt(n_train, 1)
  res_t1 <- evaluate_models(y_t1, X_train)
  
  # --- t(2) distributed errors --- #
  y_t2 <- y_true_train + rt(n_train, 2)
  res_t2 <- evaluate_models(y_t2, X_train)
  
  # --- t(3) distributed errors --- #
  y_t3 <- y_true_train + rt(n_train, 3)
  res_t3 <- evaluate_models(y_t3, X_train)
  
  
  # --- Return Vector --- #
  return(c(res_normal, res_tukey, res_t1, res_t2, res_t3))
}

# --- Execution --- #

# Detect cores (Use n-1 to keep system responsive)
num_cores <- parallel::detectCores(logical = FALSE) - 1
cl <- makeCluster(num_cores)

# Export variables to cluster
clusterExport(cl, varlist = c("n_train", "n_test", "p", "beta"))

# Run in parallel and time
runtime <- system.time({
  results_matrix <- parSapply(cl, 1:n_sim, sim_step, 
                              n_train = n_train, n_test = n_test, p = p, beta = beta)
})

stopCluster(cl)

# --- Formatting Results --- #

# Transpose matrix for ease of extraction of data
results_t <- t(results_matrix)

# Define scenarios in order they appear in results_t
scenarios <- c("Normal", "Tukey-Huber", "t1", "t2", "t3")

# Define standard column names for estimators
col_names <- c("RMSE_OLS", "RMSE_M", "RMSE_MM", "RMSE_TLS")
n_est <- length(col_names)

# Iterate, extract, and structure final dataframe
results_list <- lapply(seq_along(scenarios), function(i) {
  
  # Calculate column indices for current scenario
  cols_idx <- ((i - 1) * n_est) + 1:n_est
  
  # Extract specific columns and convert to dataframe
  data_subset <- as.data.frame(results_t[, cols_idx])
  
  # Rename columns immediately
  colnames(data_subset) <- col_names
  
  # Add metadata columns
  data_subset$Scenario <- scenarios[i]
  data_subset$Iteration <- 1:nrow(data_subset)
  
  # Reorder columns to match your preferred format
  return(data_subset[, c("Scenario", "Iteration", col_names)])
})

# Bind list into a single master Data Frame
results_df <- do.call(rbind, results_list)

# Calculate the mean RMSE for each scenario
formula_str <- paste("cbind(", paste(col_names, collapse = ", "), ") ~ Scenario")
summary_stats <- aggregate(as.formula(formula_str), data = results_df, mean)

# Function to create ratio matrix
create_ratio_matrix <- function(scenario_name, df) {
  
  # Filter row corresponding to this scenario
  subset_df <- df[df$Scenario == scenario_name, ]
  
  # Extract numeric values
  rmses <- as.numeric(subset_df[ , -1])
  
  # Re-assign names so the matrix has labels
  names(rmses) <- colnames(subset_df)[-1]
  
  # Create outer product
  ratio_mat <- outer(rmses, rmses, "/")
  
  return(round(ratio_mat, 2))
}

# Apply to data
scenarios <- unique(summary_stats$Scenario)
matrix_list <- lapply(scenarios, create_ratio_matrix, df = summary_stats)
names(matrix_list) <- scenarios

print(runtime)
print(matrix_list)
