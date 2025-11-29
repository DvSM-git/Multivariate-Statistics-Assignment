rm(list=ls())
dev.off(dev.list()["RStudioGD"])

library(robustbase)
library(parallel)

# --- SIMULATION PARAMETERS --- #
set.seed(0)
n_train <- 100
n_test <- 1000 
n_sim <- 2 # Must be greater than 1
beta <- c(0, 1) # First parameter is the intercept
p <- length(beta) - 1

# --- SINGLE SIMULATION STEP FUNCTION --- #
sim_step <- function(i, n_train, n_test, p, beta, plot = FALSE) {
  require(robustbase)
  
  # --- DATA GENERATION (COMMON STRUCTURES) --- #
  
  # Generate Design Matrices
  
  # Probability and parameters of outliers
  alpha_x <- 0.1 # Probability of being an outlier
  m_x <- 5 # Mean of the outlier distribution
  k_x <- 0.1 # Variance of the outlier distribution
  
  # Generating Tukey-Huber contaminated X for train set
  is_outlier_x <- rbinom(n_train, 1, alpha_x) 
  X_train <- matrix(rnorm(n_train * p), ncol = p) # Non-contaminated X without intercept
  X_cont <- matrix(rnorm(n_train * p, m_x, k_x), ncol = p) # Contamination
  X_combined <- X_train * (1 - is_outlier_x) + X_cont * is_outlier_x
  X_train_clean <- cbind(1, X_train) # Non-contaminated X with intercept
  X_train_cont <- cbind(1, X_combined) # Contaminated X with intercept
  
  # Generating X for train set
  X_test <- cbind(1, matrix(rnorm(n_test * p), ncol = p))
  
  # Generating y
  y_true_train_clean <- X_train_clean %*% beta # y for non-contaminated X
  y_true_train_cont <- X_train_cont %*% beta # y for contaminated x
  y_test <- X_test %*% beta # y for testing data
  
  # ---  HELPER FUNCTION FOR ESTIMATION --- #
  
  # Reduces code duplication for OLS vs Robust steps
  evaluate_models <- function(y_train, X_train) {
    # OLS
    b_ols <- .lm.fit(X_train, y_train)$coefficients
    # M-estimator
    b_M <- coef(lmrob(y_train ~ 0 + X_train))
    # MM-estimator with huber psi
    b_MM <- coef(MASS::rlm(y_train ~ 0 + X_train), method = "M", psi = psi.huber)
    # Least trimmed squares estimator
    b_lts <- coef(ltsReg(y_train ~ X_train[, -1, drop = FALSE], nsamp = "best"))
    
    # Prediction & RMSE
    rmse_ols <- sqrt(mean((y_test - (X_test %*% b_ols))^2))
    rmse_M <- sqrt(mean((y_test - (X_test %*% b_M))^2))
    rmse_MM <- sqrt(mean((y_test - (X_test %*% b_MM))^2))
    rmse_lts <- sqrt(mean((y_test - (X_test %*% b_lts))^2))
    
    return(c(rmse_ols, rmse_M, rmse_MM, rmse_lts))
  }
  
  # --- VERTICAL OUTLIER HYPERPARAMETERS --- #
  alpha_y <- 0.1 # Probability of being an outlier
  m_y <- -5 # Mean of the outlier distribution
  k_y <- 0.1 # Variance of the outlier distribution
  
  # --- NO OUTLIERS --- #
  
  # Model
  y_no <- y_true_train_clean + rnorm(n_train)
  res_no <- evaluate_models(y_no, X_train_clean)
  
  # --- VERTICAL OUTLIERS --- #
  
  # Model
  is_outlier_y <- rbinom(n_train, 1, alpha_y)
  err_vo <- rnorm(n_train) * (1 - is_outlier_y) + rnorm(n_train, m_y, k_y) * is_outlier_y
  y_vo <- y_true_train_clean + err_vo
  
  res_vo <- evaluate_models(y_vo, X_train_clean)
  
  # --- GOOD LEVERAGE POINTS --- #
  
  # Model
  y_glp<- y_true_train_cont + rnorm(n_train)
  res_glp <- evaluate_models(y_glp, X_train_cont)
  
  # --- BAD LEVERAGE POINTS --- #
  
  # Model
  is_outlier_y <- rbinom(n_train, 1, alpha_y)
  err_blp <- rnorm(n_train)
  y_blp<- y_true_train_clean + err_blp
  
  # Select those that are good leverage points
  idx_bad <- which(y_true_train_cont > 3.5)
  
  # Turn the good leverage points into bad leverage points
  y_blp <- y_true_train_cont
  y_blp[idx_bad] <- y_blp[idx_bad] + 2*m_y
  
  y_blp <- y_blp + rnorm(n_train)
  
  res_blp <- evaluate_models(y_blp, X_train_cont)
  

  # --- PLOTTING SCATTER PLOTS  --- #
  if (plot == TRUE) {
    
    # Set up the plots in a 2*2 grid
    par(mfrow = c(2, 2))
    
    # Plot graphs 
    plot(X_train_clean[, -c(1)], y_no, xlab = "", ylab = "", xaxt = "n", yaxt = "n", main = "No Outlier")
    plot(X_train_clean[, -c(1)], y_vo, xlab = "", ylab = "", xaxt = "n", yaxt = "n", main = "Vertical Outlier")
    plot(X_train_cont[, -c(1)], y_glp, xlab = "", ylab = "", xaxt = "n", yaxt = "n", main = "Good Leverage Point")
    plot(X_train_cont[, -c(1)], y_blp, xlab = "", ylab = "", xaxt = "n", yaxt = "n", main = "Bad Leverage Point")

  }
    
  
  # --- RETURN VECTOR --- #
  return(c(res_no, res_vo, res_glp, res_blp))
}

# --- EXECUTION --- #

# Dummy run to plot the plots
dummy_run <- sim_step(i = 1, n_train = n_train, n_test = n_test, p = p, beta = beta, plot = TRUE)

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

# --- FORMATTING RESULTS --- #

# Transpose matrix for ease of extraction of data
results_t <- t(results_matrix)

# Define scenarios in order they appear in results_t
scenarios <- c("No Outliers", "Vertical Outliers", "Good Leverage Points", "Bad Leverage Points")

# Define standard column names for estimators
col_names <- c("RMSE_OLS", "RMSE_M", "RMSE_MM", "RMSE_lts")
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

# --- PRINTING RESULTS --- #
print(runtime)
print(matrix_list)
