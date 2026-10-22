# Robyn fit on the same weekly series as the PyMC manifest.
# This is a bake-off participant. It is not a production model.
# Usage: Rscript robyn/run_robyn.R path/to/dt_simulated_weekly.csv path/to/dt_prophet_holidays_de.csv path/to/manifest.json

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Need weekly csv, holiday csv, and the PyMC manifest.json")
}

library(Robyn)
library(jsonlite)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) {
  stop("Run this file with Rscript so the next-ruble operator can be sourced.")
}
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
source(file.path(script_dir, "next_ruble.R"))

weekly <- read.csv(args[[1]], stringsAsFactors = FALSE)
weekly <- weekly[order(as.Date(weekly$DATE)), ]
holidays <- read.csv(args[[2]], stringsAsFactors = FALSE)
holidays$country <- "DE"
holidays$year <- as.integer(format(as.Date(holidays$ds), "%Y"))
manifest <- fromJSON(args[[3]])

paid_spend <- c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S")
paid_media <- c("tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P")
hyper <- list()
for (media in c(paid_media, "newsletter")) {
  hyper[[paste0(media, "_alphas")]] <- c(0.5, 3)
  hyper[[paste0(media, "_gammas")]] <- c(0.3, 1)
  hyper[[paste0(media, "_thetas")]] <- c(0, 0.3)
}
train_share <- (nrow(weekly) - manifest$holdout_weeks) / nrow(weekly)
hyper$train_size <- c(train_share, train_share)

input <- robyn_inputs(
  dt_input = weekly,
  dt_holidays = holidays,
  date_var = "DATE",
  dep_var = "revenue",
  dep_var_type = "revenue",
  prophet_vars = c("trend", "season", "holiday"),
  prophet_country = "DE",
  context_vars = c("competitor_sales_B", "events"),
  paid_media_spends = paid_spend,
  paid_media_vars = paid_media,
  organic_vars = "newsletter",
  factor_vars = "events",
  window_start = manifest$date_min,
  window_end = manifest$date_max,
  adstock = "geometric",
  hyperparameters = hyper
)

# Five trials is Robyn's own minimum for an initial model. One trial of 2000
# iterations takes about three minutes on one core, so 2000 trials does not finish.
trials <- as.integer(Sys.getenv("ROBYN_TRIALS", "5"))
iterations <- as.integer(Sys.getenv("ROBYN_ITERS", "2000"))
set.seed(manifest$seed)
output <- robyn_run(
  InputCollect = input,
  iterations = iterations,
  trials = trials,
  cores = 1,
  outputs = FALSE,
  ts_validation = TRUE
)
chosen <- robyn_outputs(
  InputCollect = input,
  OutputModels = output,
  select_model = "clusters",
  export = FALSE,
  plot_pareto = FALSE
)
if (is.null(chosen$xDecompVec) || is.null(chosen$resultHypParam) || is.null(chosen$xDecompAgg)) {
  stop("Unexpected Robyn output. Found: ", paste(names(chosen), collapse = ", "))
}
winners <- chosen$resultHypParam[chosen$resultHypParam$top_sol %in% TRUE, ]
if (nrow(winners) == 0) {
  stop("Robyn returned no top_sol winner.")
}
winner <- winners$solID[which.min(winners$nrmse_test)]
if (length(winner) != 1 || is.na(winner)) {
  stop("Robyn did not choose one winner.")
}
channel_of <- c(
  tv_S = "tv",
  ooh_S = "ooh",
  print_S = "print",
  facebook_I = "facebook",
  search_clicks_P = "search"
)
if (!("coef" %in% names(chosen$xDecompAgg))) {
  stop("Robyn decomposition has no coef column. Found: ", paste(names(chosen$xDecompAgg), collapse = ", "))
}
n_train <- nrow(weekly) - as.integer(manifest$holdout_weeks)
horizon <- 8L
if (n_train <= horizon) {
  stop("Training window is shorter than the adstock horizon.")
}
spend_of <- c(
  tv_S = "tv_S",
  ooh_S = "ooh_S",
  print_S = "print_S",
  facebook_I = "facebook_S",
  search_clicks_P = "search_S"
)
marginal_for <- function(sol) {
  hyp <- chosen$resultHypParam[chosen$resultHypParam$solID == sol, ]
  agg <- chosen$xDecompAgg[chosen$xDecompAgg$solID == sol, ]
  if (nrow(hyp) != 1) {
    stop("Model ", sol, " is not one hyperparameter row.")
  }
  vapply(names(channel_of), function(media_name) {
    theta_name <- paste0(media_name, "_thetas")
    alpha_name <- paste0(media_name, "_alphas")
    gamma_name <- paste0(media_name, "_gammas")
    for (column in c(theta_name, alpha_name, gamma_name)) {
      if (!(column %in% names(hyp))) {
        stop("Model ", sol, " is missing ", column)
      }
    }
    theta <- as.numeric(hyp[[theta_name]])
    alpha <- as.numeric(hyp[[alpha_name]])
    gamma <- as.numeric(hyp[[gamma_name]])
    coef <- as.numeric(agg$coef[agg$rn == media_name])
    if (length(coef) != 1 || !is.finite(coef) || !is.finite(theta) || !is.finite(alpha) || !is.finite(gamma)) {
      stop("Coefficients are not finite for ", media_name, " in ", sol)
    }
    media <- as.numeric(weekly[[media_name]][seq_len(n_train)])
    spend <- as.numeric(weekly[[spend_of[[media_name]]]][seq_len(n_train)])
    next_ruble_roi(media, spend, theta, alpha, gamma, coef, horizon = horizon)
  }, numeric(1))
}
marginal <- marginal_for(winner)
roi <- data.frame(
  channel = unname(channel_of[names(marginal)]),
  marginal_roi_next_ruble = as.numeric(marginal),
  stringsAsFactors = FALSE
)
# The spread is across the first Pareto front of this run. It is not a posterior.
front <- unique(chosen$resultHypParam$solID[chosen$resultHypParam$robynPareto %in% 1])
front <- front[front %in% chosen$xDecompAgg$solID]
if (length(front) < 2) {
  stop("The first Pareto front has fewer than two models, so there is no spread.")
}
front_roi <- t(vapply(front, marginal_for, numeric(length(channel_of))))
colnames(front_roi) <- unname(channel_of)
roi_low <- apply(front_roi, 2, stats::quantile, probs = 0.025, names = FALSE)
roi_high <- apply(front_roi, 2, stats::quantile, probs = 0.975, names = FALSE)
decomp <- chosen$xDecompVec[chosen$xDecompVec$solID == winner, ]
decomp <- decomp[order(as.Date(decomp$ds)), ]
decomp <- decomp[!duplicated(decomp$ds), ]
dates <- unique(decomp$ds)
if (length(dates) < manifest$holdout_weeks) {
  stop("Decomposition has fewer dates than the holdout.")
}
hold_dates <- utils::tail(dates, manifest$holdout_weeks)
hold <- decomp[decomp$ds %in% hold_dates, ]
if (nrow(hold) != manifest$holdout_weeks) {
  stop("Holdout is not one row per date.")
}
holdout_mape <- mean(abs(hold$dep_var - hold$depVarHat) / pmax(abs(hold$dep_var), 1))
root <- dirname(script_dir)
artifact_dir <- file.path(root, "artifacts")
dir.create(artifact_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(roi, file.path(artifact_dir, "robyn_roi.csv"), row.names = FALSE)
export <- list(
  model = "robyn",
  manifest = manifest,
  estimand = "marginal_roi_next_ruble",
  holdout_mape = holdout_mape,
  holdout_actual = as.numeric(hold$dep_var),
  holdout_pred = as.numeric(hold$depVarHat),
  channel_roi = setNames(as.list(roi$marginal_roi_next_ruble), roi$channel),
  channel_roi_low = as.list(roi_low),
  channel_roi_high = as.list(roi_high),
  interval = paste0("2.5th to 97.5th percentile across ", length(front), " first-Pareto-front models, not a posterior; the point is the selected model and can sit outside it"),
  interval_kind = "model_spread",
  r_hat_max = NULL,
  prior_sensitivity_relative = NULL,
  trials = trials,
  selected_model = winner
)
write_json(export, file.path(artifact_dir, "robyn_export.json"), auto_unbox = TRUE, pretty = TRUE, null = "null")
message("Wrote ", file.path(artifact_dir, "robyn_export.json"), " for ", winner, " with holdout MAPE ", signif(holdout_mape, 4))
