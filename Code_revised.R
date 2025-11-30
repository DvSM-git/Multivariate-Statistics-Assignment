rm(list=ls())
if (!is.null(dev.list())) dev.off()

library(robustbase)
library(parallel)
library(MASS)
library(ggplot2)
library(tidyr)
library(dplyr)
library(ggplot2)


# --- DATA GENERATION --- #
generate_data <- function(n, p, beta, type, params) {
  # Base X generation (Clean and Contaminated)
  X_base <- matrix(rnorm(n * p), ncol = p)
  X_cont <- matrix(rnorm(n * p, params$x$mean, params$x$sd), ncol = p)
  
  # Outlier mask for X
  idx_x <- rbinom(n, 1, params$x$alpha) == 1
  X_combined <- X_base; X_combined[idx_x, ] <- X_cont[idx_x, ]
  
  X_cln <- cbind(1, X_base)
  X_mix <- cbind(1, X_combined)
  
  # Base Y components
  y_true_cln <- X_cln %*% beta
  y_true_mix <- X_mix %*% beta
  
  # Vertical error generation
  idx_y <- rbinom(n, 1, params$y$alpha) == 1
  err_vo <- rnorm(n); err_vo[idx_y] <- rnorm(sum(idx_y), params$y$mean, params$y$sd)
  
  # Scenario Logic
  if (type == "No Outliers") {
    return(list(y = y_true_cln + rnorm(n), X = X_cln))
    
  } else if (type == "Vertical Outliers") {
    return(list(y = y_true_cln + err_vo, X = X_cln))
    
  } else if (type == "Good Leverage Points") {
    return(list(y = y_true_mix + rnorm(n), X = X_mix))
    
  } else if (type == "Bad Leverage Points") {
    # Specific BLP logic from original code
    y_blp <- y_true_mix
    idx_bad <- which(y_true_mix > 3.5)
    y_blp[idx_bad] <- y_blp[idx_bad] + 2 * params$y$mean
    return(list(y = y_blp + rnorm(n), X = X_mix))
  }
}

# --- ESTIMATION & EVALUATION --- #
fit_and_evaluate <- function(y_train, X_train, X_test, y_test) {
  # Estimation
  fits <- list(
    OLS = .lm.fit(X_train, y_train)$coefficients,
    MM = tryCatch(coef(lmrob(y_train ~ 0 + X_train)), error = function(e) rep(NA, ncol(X_train))),
    M = coef(MASS::rlm(y_train ~ 0 + X_train, psi = psi.huber, maxit = 100)),
    lts = coef(ltsReg(y_train ~ X_train[, -1, drop = FALSE], nsamp = "best"))
  )
  
  # Prediction & Loss
  sapply(fits, function(b) {
    if (any(is.na(b))) return(rep(NA, 4))
    e <- as.numeric(y_test - X_test %*% b)
    c(RMSE  = sqrt(mean(e^2)),
      MAE   = mean(abs(e)),
      MedAE = median(abs(e)),
      Huber = mean(ifelse(abs(e) < 1.345, 0.5 * e^2, 1.345 * (abs(e) - 0.5 * 1.345))))
  })
}

# --- MAIN SIMULATION STEP --- #
sim_step <- function(i, cfg) {
  # Generate Test Data (Common for all scenarios in this step)
  X_test <- cbind(1, matrix(rnorm(cfg$n_test * (length(cfg$beta)-1)), ncol = length(cfg$beta)-1))
  y_test <- X_test %*% cfg$beta
  
  scenarios <- c("No Outliers", "Vertical Outliers", "Good Leverage Points", "Bad Leverage Points")
  
  # Iterate over scenarios and flatten results
  res <- lapply(scenarios, function(scen) {
    dat <- generate_data(cfg$n_train, length(cfg$beta)-1, cfg$beta, scen, cfg$params)
    metrics <- fit_and_evaluate(dat$y, dat$X, X_test, y_test)
    as.vector(metrics) # Flatten 4x4 matrix to vector
  })
  
  return(unlist(res))
}

# --- PLOT OF SENSITIVITY CURVES --- #
plot_breakdown_curves <- function(breakdown_df, metric = "RMSE") {
  require(ggplot2)
  require(dplyr)
  require(tidyr)
  
  # Select and Reshape Data
  plot_data <- breakdown_df %>%
    dplyr::select(Scenario, Alpha, starts_with(paste0(metric, "_"))) %>%
    tidyr::pivot_longer(
      cols = starts_with(paste0(metric, "_")),
      names_to = "Estimator",
      names_prefix = paste0(metric, "_"), # Removes "RMSE_" from legend
      values_to = "Value"
    )
  
  # Generate Plot
  p <- ggplot(plot_data, aes(x = Alpha, y = Value, color = Estimator)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    facet_wrap(~Scenario, scales = "free_y") +
    scale_color_brewer(palette = "Set1") +
    theme_bw() +
    labs(
      title = paste0("Empirical Breakdown Curves: ", metric),
      subtitle = "Performance degradation as contamination increases",
      x = "Contamination Level (Alpha)",
      y = paste("Mean", metric)
    ) +
    theme(
      legend.position = "bottom",
      strip.text = element_text(face = "bold"),
      plot.title = element_text(face = "bold")
    )
  
  return(p)
}

# --- EXECUTION --- #

# Define Contamination Sequence (0% to 45%)
alpha_seq <- seq(0, 0.45, by = 0.05)
breakdown_history <- list()

# Configuration
set.seed(0)
config <- list(
  n_train = 100,
  n_test = 1000,
  n_sim = 1000,
  beta = c(0, 1),
  params  = list(
    x = list(alpha = 0, mean = 5,  sd = 0.1),
    y = list(alpha = 0, mean = -5, sd = 0.1)
  )
)

cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("generate_data", "fit_and_evaluate", "lmrob", 
                    "ltsReg", "rlm", "psi.huber", "config")) 

for (curr_alpha in alpha_seq) {

  # Update Config
  config$params$x$alpha <- curr_alpha
  config$params$y$alpha <- curr_alpha
  
  p <- length(config$beta) - 1
  
  # Run Simulation
  clusterExport(cl, "config") 
  results_matrix <- parSapply(cl, 1:config$n_sim, sim_step, cfg = config)

  # --- RESULTS AGGREGATION --- #
  process_results <- function(res_mat, scenarios) {
    metrics <- c("RMSE", "MAE", "MedAE", "Huber")
    ests <- c("OLS", "MM", "M", "lts")
    n_metrics_per_scen <- length(metrics) * length(ests)
    
    # Reshape to long format
    df_list <- lapply(seq_along(scenarios), function(i) {
      idx <- ((i - 1) * n_metrics_per_scen) + 1:n_metrics_per_scen
      mat <- t(res_mat[idx, ])
      colnames(mat) <- as.vector(outer(metrics, ests, paste, sep = "_"))
      data.frame(Scenario = scenarios[i], mat)
    })
    
    do.call(rbind, df_list)
  }
  
  results_df <- process_results(results_matrix, c("No Outliers", "Vertical Outliers", "Good Leverage Points", "Bad Leverage Points"))
  
  # Add calculation to sensitivity curve list
  summary_stats <- aggregate(. ~ Scenario, data = results_df, mean)
  summary_stats$Alpha <- curr_alpha
  breakdown_history[[as.character(curr_alpha)]] <- summary_stats
  

  # Function to print ratios
  print_ratios <- function(df, metric) {
    cat(paste0("\n=== ", metric, " Ratio Matrices ===\n"))
    lapply(split(df, df$Scenario), function(sub) {
      cols <- grep(paste0("^", metric, "_"), names(sub))
      vals <- as.numeric(sub[1, cols])
      names(vals) <- c("OLS", "M", "MM", "lts")
      round(outer(vals, vals, "/"), 2)
    }) |> print()
  }
  
  # --- PLOT OF ESTIMATED LINES VS REAL LINE --- #
  visualise_scenarios <- function(cfg) {
    par(mfrow = c(2, 2))
    
    x_grid <- cbind(1, seq(-4, 8, length.out = 200))
    y_true <- x_grid %*% cfg$beta
    
    scenarios <- c("No Outliers", "Vertical Outliers", "Good Leverage Points", "Bad Leverage Points")
    
    invisible(lapply(scenarios, function(scen) {
      dat <- generate_data(cfg$n_train, length(cfg$beta)-1, cfg$beta, scen, cfg$params)
      
      # Re-estimate for plotting lines
      fits <- list(
        OLS = .lm.fit(dat$X, dat$y)$coefficients,
        MM   = coef(lmrob(dat$y ~ 0 + dat$X)),
        M  = coef(MASS::rlm(dat$y ~ 0 + dat$X, psi = psi.huber, maxit = 100)),
        lts = coef(ltsReg(dat$y ~ dat$X[, -1, drop = FALSE], nsamp = "best"))
      )
      
      # Plot
      plot(x_grid[,2], y_true, type = 'l', lwd = 2, main = scen, xlab = "x", ylab = "y", ylim = range(dat$y))
      points(dat$X[,2], dat$y, pch = 16, cex = 0.6, col = "grey50")
      
      cols <- c("red", "blue", "darkgreen", "purple")
      for (j in seq_along(fits)) lines(x_grid[,2], x_grid %*% fits[[j]], col = cols[j], lwd = 1.5)
      
      if (scen == "No Outliers") legend("topleft", names(fits), col = cols, lty = 1, bty = "n", cex = 0.7)
    }))
  }
  
  # --- PLOT OF DISTRIBUTION OF AVERAGE ERROR FUNCTIONS --- #
  plot_all_distributions <- function(df, contamination) {
    
    # Transform data to fully long format
    long_df <- df %>%
      pivot_longer(
        cols = -Scenario,
        names_to = c("Metric", "Estimator"),
        names_sep = "_",              # Split "RMSE_OLS" into "RMSE" and "OLS"
        values_to = "Value"
      )
    
    # Get list of unique metrics (RMSE, MAE, MedAE, Huber)
    metrics_list <- unique(long_df$Metric)
    
    # Loop through metrics and print one plot per metric
    for (met in metrics_list) {
      
      # Filter data for the current metric only
      plot_data <- long_df %>% filter(Metric == met)
      
      p <- ggplot(plot_data, aes(x = Estimator, y = Value, fill = Estimator)) +
        geom_boxplot(outlier.size = 0.5, alpha = 0.6) +
        
        # Facet by Scenario with independent y-axes
        facet_wrap(~Scenario, scales = "free_y") + 
        
        theme_bw() +
        labs(
          title = paste0(met, " Distribution (Contamination: ", contamination, ")"),
          y = met,
          x = "Estimator"
        ) +
        theme(
          legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      print(p)
    }
  }
  
  # --- OUTPUT --- #
  visualise_scenarios(config)
  plot_all_distributions(results_df, curr_alpha)
  print(paste0("==================== ", curr_alpha * 100, "% Contamination Level===================="))
  for (m in c("RMSE", "MAE", "MedAE",  "Huber")) print_ratios(summary_stats, m)
}

stopCluster(cl)

final_breakdown_df <- do.call(rbind, breakdown_history)
print(plot_breakdown_curves(final_breakdown_df, metric = "RMSE"))