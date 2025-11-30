rm(list=ls())
if (!is.null(dev.list())) dev.off(dev.list()["RStudioGD"])

library(robustbase)
library(parallel)
library(MASS)

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
    
    # MM-estimator with Huber psi
    b_MM <- coef(MASS::rlm(y_train ~ 0 + X_train, method = "M", psi = MASS::psi.huber, maxit = 100))
    
    # LTS estimator (max breakdown alpha from slide)
    n <- length(y_train)
    alpha <- (floor(n/2) + floor((p + 1)/2)) / n
    b_lts <- ltsReg(x = X_train[, -1, drop = FALSE],
                    y = y_train,
                    nsamp = "best",
                    alpha = alpha)$coefficients
    
    # Predictions
    preds <- list(
      OLS = X_test %*% b_ols,
      M   = X_test %*% b_M,
      MM  = X_test %*% b_MM,
      lts = X_test %*% b_lts
    )
    
    # Losses
    mae      <- function(e) mean(abs(e))
    medae    <- function(e) median(abs(e))
    hub_loss <- function(e, delta = 1.345)
      ifelse(abs(e) < delta, 0.5 * e^2,
             delta * (abs(e) - 0.5 * delta))
    huber_risk <- function(e, delta = 1.345) mean(hub_loss(e, delta))
    rmse     <- function(e) sqrt(mean(e^2))
    
    loss_names <- c("RMSE", "MAE", "MedAE", "Huber")
    est_names  <- names(preds)
    
    loss_mat <- matrix(NA_real_,
                       nrow = length(loss_names),
                       ncol = length(est_names),
                       dimnames = list(loss_names, est_names))
    
    for (j in seq_along(preds)) {
      e <- as.numeric(y_test - preds[[j]])
      loss_mat["RMSE",  j] <- rmse(e)
      loss_mat["MAE",   j] <- mae(e)
      loss_mat["MedAE", j] <- medae(e)
      loss_mat["Huber", j] <- huber_risk(e)
    }
    
    loss_mat
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
  #is_outlier_y <- rbinom(n_train, 1, alpha_y) #delete
  #err_blp <- rnorm(n_train) #delete
  #y_blp<- y_true_train_clean + err_blp #delete
  
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
  # Each res_* is a 4x4 matrix (loss x estimator)
  return(c(as.vector(res_no),
           as.vector(res_vo),
           as.vector(res_glp),
           as.vector(res_blp)))
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

# 
losses     <- c("RMSE", "MAE", "MedAE", "Huber")
estimators <- c("OLS", "M", "MM", "lts")

# matrix with rows = losses, cols = estimators, vectorised column-wise
metric_names_mat <- outer(losses, estimators, paste, sep = "_")
metric_names <- as.vector(metric_names_mat)
metric_names
n_est <- length(metric_names)  # 16

# Define scenarios in order they appear in results_t
scenarios <- c("No Outliers", "Vertical Outliers", "Good Leverage Points", "Bad Leverage Points")

results_list <- lapply(seq_along(scenarios), function(i) {
  
  # Calculate column indices for current scenario
  cols_idx <- ((i - 1) * n_est) + 1:n_est
  
  # Extract and name
  data_subset <- as.data.frame(results_t[, cols_idx])
  colnames(data_subset) <- metric_names
  
  data_subset$Scenario <- scenarios[i]
  data_subset$Iteration <- 1:nrow(data_subset)
  
  data_subset[, c("Scenario", "Iteration", metric_names)]
})


# Bind list into a single master Data Frame
results_df <- do.call(rbind, results_list)

# Calculate the mean RMSE for each scenario
formula_str <- paste("cbind(", paste(metric_names, collapse = ", "), ") ~ Scenario")
summary_stats <- aggregate(as.formula(formula_str), data = results_df, median)

create_ratio_matrix <- function(scenario_name, metric, df) {
  # columns for that metric: RMSE_OLS, RMSE_M, ...
  cols_idx <- grep(paste0("^", metric, "_"), colnames(df))
  
  subset_df <- df[df$Scenario == scenario_name, cols_idx, drop = FALSE]
  
  vals <- as.numeric(subset_df)
  names(vals) <- sub(paste0(metric, "_"), "", colnames(subset_df))  # OLS, M, MM, lts
  
  ratio_mat <- outer(vals, vals, "/")
  round(ratio_mat, 2)
}

scens <- unique(summary_stats$Scenario)

rmse_ratio_list  <- lapply(scens, create_ratio_matrix, metric = "RMSE", df = summary_stats)
mae_ratio_list   <- lapply(scens, create_ratio_matrix, metric = "MAE", df = summary_stats)
huber_ratio_list <- lapply(scens, create_ratio_matrix, metric = "Huber", df = summary_stats)

names(rmse_ratio_list) <- scens
names(mae_ratio_list)   <- scens
names(huber_ratio_list) <- scens

# --- PRINTING RESULTS --- #
print(runtime)

cat("\n=== RMSE ratio matrices ===\n")
print(rmse_ratio_list)

cat("\n=== MAE ratio matrices ===\n")
print(mae_ratio_list)

cat("\n=== Huber ratio matrices ===\n")
print(huber_ratio_list)

# --- VISUALISATION (LIKE SLIDE 65) --- #

visualise_scenarios <- function() {
  # Use same global parameters as in the simulation
  # n_train, p, beta must exist in the global environment
  
  set.seed(123)  # for reproducibility of the visualisation
  
  # --- 1. Generate X (same structure as in sim_step) --- #
  alpha_x <- 0.1
  m_x     <- 5
  k_x     <- 0.1
  
  is_outlier_x <- rbinom(n_train, 1, alpha_x)
  X_train      <- matrix(rnorm(n_train * p), ncol = p)
  X_cont       <- matrix(rnorm(n_train * p, m_x, k_x), ncol = p)
  X_combined   <- X_train * (1 - is_outlier_x) + X_cont * is_outlier_x
  
  # With intercept
  X_train_clean <- cbind(1, X_train)
  X_train_cont  <- cbind(1, X_combined)
  
  # True mean structure
  y_true_train_clean <- X_train_clean %*% beta
  y_true_train_cont  <- X_train_cont %*% beta
  
  # --- 2. Generate y for the four scenarios --- #
  alpha_y <- 0.1
  m_y     <- -5
  k_y     <- 0.1
  
  ## (a) No outliers
  y_no <- y_true_train_clean + rnorm(n_train)
  
  ## (b) Vertical outliers
  is_outlier_y <- rbinom(n_train, 1, alpha_y)
  err_vo <- rnorm(n_train) * (1 - is_outlier_y) +
    rnorm(n_train, m_y, k_y) * is_outlier_y
  y_vo <- y_true_train_clean + err_vo
  
  ## (c) Good leverage points
  y_glp <- y_true_train_cont + rnorm(n_train)
  
  ## (d) Bad leverage points
  err_blp <- rnorm(n_train)
  y_blp   <- y_true_train_clean + err_blp
  idx_bad <- which(y_true_train_cont > 3.5)
  y_blp   <- y_true_train_cont
  y_blp[idx_bad] <- y_blp[idx_bad] + 2 * m_y
  y_blp <- y_blp + rnorm(n_train)
  
  # --- 3. Helper: fit all four models and return coefficients --- #
  fit_models <- function(y, X) {
    # OLS (matrix interface, X already has intercept)
    b_ols <- .lm.fit(X, y)$coefficients
    
    # M-estimator (lmrob) – using 0 + X because X already includes intercept
    b_M   <- coef(lmrob(y ~ 0 + X))
    
    # MM-estimator (Huber, via rlm) – more iterations for stability
    fit_MM <- MASS::rlm(y ~ 0 + X, psi = psi.huber, maxit = 100)
    b_MM   <- coef(fit_MM)
    
    # LTS estimator – X without intercept column, ltsReg adds its own intercept
    b_lts  <- coef(ltsReg(y ~ X[, -1, drop = FALSE], nsamp = "best"))
    
    list(OLS = b_ols, M = b_M, MM = b_MM, LTS = b_lts)
  }
  
  coefs_no  <- fit_models(y_no,  X_train_clean)
  coefs_vo  <- fit_models(y_vo,  X_train_clean)
  coefs_glp <- fit_models(y_glp, X_train_cont)
  coefs_blp <- fit_models(y_blp, X_train_cont)
  
  # --- 4. Grid of x-values for plotting the lines --- #
  # Use a range that covers typical X values and some leverage
  x_grid <- seq(-4, 8, length.out = 200)
  X_grid <- cbind(1, x_grid)
  
  # True 45-degree line: y = x (because beta = c(0, 1))
  y_true <- as.numeric(X_grid %*% beta)
  
  # --- 5. Plot function for one scenario --- #
  plot_scenario <- function(coefs, main_title) {
    # "Scatter" = points on the true line (as per your idea)
    plot(x_grid, y_true,
         pch = 16, cex = 0.4,
         xlab = "x", ylab = "y",
         main = main_title)
    
    # True line
    lines(x_grid, y_true, col = "black", lwd = 2)
    
    # Fitted lines
    lines(x_grid, as.numeric(X_grid %*% coefs$OLS), col = "red",       lwd = 2)
    lines(x_grid, as.numeric(X_grid %*% coefs$M),   col = "blue",      lwd = 2)
    lines(x_grid, as.numeric(X_grid %*% coefs$MM),  col = "darkgreen", lwd = 2)
    lines(x_grid, as.numeric(X_grid %*% coefs$LTS), col = "purple",    lwd = 2)
    
    legend("topleft",
           legend = c("True", "OLS", "M", "MM", "LTS"),
           col    = c("black", "red", "blue", "darkgreen", "purple"),
           lwd    = 2,
           bty    = "n",
           cex    = 0.8)
  }
  
  # --- 6. 2x2 layout and draw all scenarios --- #
  op <- par(mfrow = c(2, 2))
  on.exit(par(op), add = TRUE)
  
  plot_scenario(coefs_no,  "No Outliers")
  plot_scenario(coefs_vo,  "Vertical Outliers")
  plot_scenario(coefs_glp, "Good Leverage Points")
  plot_scenario(coefs_blp, "Bad Leverage Points")
}

# Call this *after* the simulation or standalone:
visualise_scenarios()
