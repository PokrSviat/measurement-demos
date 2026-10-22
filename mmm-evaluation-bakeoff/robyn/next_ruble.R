# Spend-weighted finite difference of one extra ruble.
# Hill inflexion stays on the unbumped series. The sum is that week and its
# horizon only, and the week counts only when that horizon sits inside the series.
# Robyn's saturation_hill(..., x_marginal=) is that freeze: inflexion is max(x) * gamma.

next_ruble_roi <- function(media, spend, theta, alpha, gamma, coef, horizon = 8L) {
  media <- as.numeric(media)
  spend <- as.numeric(spend)
  horizon <- as.integer(horizon)
  if (length(media) != length(spend)) {
    stop("Spend and media must share a week index.")
  }
  if (horizon < 1L || length(media) <= horizon) {
    stop("Need a week with a complete horizon.")
  }
  if (!is.finite(theta) || !is.finite(alpha) || !is.finite(gamma) || !is.finite(coef)) {
    stop("Coefficients must be finite.")
  }
  robyn_ns <- asNamespace("Robyn")
  decayed <- robyn_ns$transform_adstock(media, adstock = "geometric", theta = theta)$x_decayed
  base <- coef * robyn_ns$saturation_hill(decayed, alpha = alpha, gamma = gamma)$x_saturated
  last <- length(media) - horizon + 1L
  usable <- which(spend[seq_len(last)] > 0)
  if (length(usable) == 0) {
    stop("No positive-spend week has a complete horizon.")
  }
  deltas <- vapply(usable, function(week) {
    bumped <- media
    bumped[[week]] <- bumped[[week]] + media[[week]] / spend[[week]]
    bumped_decayed <- robyn_ns$transform_adstock(bumped, adstock = "geometric", theta = theta)$x_decayed
    changed <- coef * robyn_ns$saturation_hill(
      decayed,
      alpha = alpha,
      gamma = gamma,
      x_marginal = bumped_decayed
    )$x_saturated
    end <- week + horizon - 1L
    sum(changed[week:end] - base[week:end])
  }, numeric(1))
  stats::weighted.mean(deltas, spend[usable])
}
