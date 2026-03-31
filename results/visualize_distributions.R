#!/usr/bin/env Rscript
# visualize_distributions.R
#
# Distributional visualization suite for platform-ABM analysis rework.
# Produces four figures:
#   1. slope_distribution.pdf   — violin/strip plot of escalation slopes across factorial
#   2. spaghetti_epoch.pdf      — superposed epoch with individual platform traces
#   3. burst_marginals.pdf      — burst size marginal distributions (3x3 grid)
#   4. enclave_settling.pdf     — enclave settling time histograms
#
# Usage:
#   Rscript results/visualize_distributions.R
#   (run from project root)
#
# Inputs:
#   results/distributions/escalation_slopes.csv
#   results/distributions/burst_sizes.csv
#   results/distributions/enclave_metrics.csv
#   results/exp2/exp2_np27_rho015_alpha10/per_iter_burst_analysis.json (spaghetti)
#
# Outputs saved to results/figures/ (created if needed).

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(forcats)
  library(scales)
})

# --- Paths ---
dist_dir  <- "results/distributions"
fig_dir   <- "results/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ===========================================================================
# Figure 1: Slope distribution violin/strip plot
# ===========================================================================

message("Figure 1: slope_distribution.pdf ...")

slopes_path <- file.path(dist_dir, "escalation_slopes.csv")

if (!file.exists(slopes_path)) {
  message("  WARNING: ", slopes_path, " not found — skipping figure 1")
} else {
  slopes <- read.csv(slopes_path, stringsAsFactors = FALSE)

  # Compute fraction with n_bursts >= 3 per cell (already filtered, but annotate cell counts)
  slopes <- slopes %>%
    mutate(
      n_platforms_label = paste0("N[p] == ", n_platforms),
      alpha_label        = factor(paste0("alpha == ", alpha),
                                  levels = paste0("alpha == ", sort(unique(alpha)))),
      alpha_fac          = factor(alpha, levels = sort(unique(alpha)))
    )

  # Annotation: count per facet x color cell
  cell_counts <- slopes %>%
    group_by(n_platforms, alpha) %>%
    summarise(
      n_rows        = n(),
      n_platforms_label = first(n_platforms_label),
      alpha_fac     = first(alpha_fac),
      .groups       = "drop"
    ) %>%
    mutate(
      label = paste0("n=", n_rows),
      slope = max(slopes$slope, na.rm = TRUE) * 1.05
    )

  p1 <- ggplot(slopes, aes(x = alpha_fac, y = slope, fill = alpha_fac, colour = alpha_fac)) +
    geom_violin(alpha = 0.35, trim = FALSE, scale = "width") +
    geom_jitter(alpha = 0.3, size = 0.5, width = 0.2, shape = 16) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40", linewidth = 0.4) +
    facet_wrap(~ n_platforms_label, labeller = label_parsed, ncol = 3) +
    scale_fill_brewer(palette = "Set1", name = expression(alpha)) +
    scale_colour_brewer(palette = "Set1", name = expression(alpha)) +
    labs(
      title    = "Distribution of Per-Platform Escalation Slopes",
      subtitle = "Platforms with \u2265 3 bursts only; strip points overlaid (alpha=0.3)",
      x        = expression(alpha ~ "(vampirism intensity)"),
      y        = "Escalation slope (burst size per burst)"
    ) +
    theme_bw(base_size = 11) +
    theme(
      strip.background = element_rect(fill = "grey90"),
      legend.position  = "bottom"
    )

  ggsave(
    file.path(fig_dir, "slope_distribution.pdf"),
    p1,
    width = 7, height = 5, dpi = 300
  )
  message("  Saved slope_distribution.pdf")
}

# ===========================================================================
# Figure 2: Spaghetti superposed epoch
#
# Uses per_iter_burst_analysis.json for exp2_np27_rho015_alpha10.
# Reads burst_steps per platform-iteration, aligns to event number (not
# calendar step), and plots individual grey traces with bold red mean.
# ===========================================================================

message("Figure 2: spaghetti_epoch.pdf ...")

spaghetti_json <- "results/exp2/exp2_np27_rho015_alpha10/per_iter_burst_analysis.json"

if (!file.exists(spaghetti_json)) {
  message("  WARNING: ", spaghetti_json, " not found — skipping figure 2")
} else {
  # jsonlite for JSON parsing
  suppressPackageStartupMessages(library(jsonlite))

  raw_text   <- readLines(spaghetti_json, warn = FALSE)
  raw_text   <- gsub("\\bNaN\\b", "null", raw_text)
  burst_json <- fromJSON(paste(raw_text, collapse = "\n"), simplifyVector = FALSE)

  # Build long data frame: one row per (iteration, platform, event_number)
  spag_rows <- list()
  for (iter_key in names(burst_json)) {
    iter_val <- as.integer(iter_key)
    platforms_data <- burst_json[[iter_key]]
    for (pid in names(platforms_data)) {
      pdata <- platforms_data[[pid]]
      bsteps <- pdata$burst_steps
      if (is.null(bsteps) || length(bsteps) == 0) next
      bsteps_vec <- unlist(bsteps)
      for (ev_num in seq_along(bsteps_vec)) {
        spag_rows[[length(spag_rows) + 1]] <- list(
          iteration     = iter_val,
          platform_id   = pid,
          event_number  = ev_num,
          calendar_step = bsteps_vec[ev_num]
        )
      }
    }
  }

  if (length(spag_rows) == 0) {
    message("  WARNING: no burst events found in JSON — skipping figure 2")
  } else {
    spag_df <- bind_rows(spag_rows)

    # Trace identifier: one per (iteration, platform)
    spag_df <- spag_df %>%
      mutate(trace_id = paste0(iteration, "_", platform_id))

    # Mean trace: mean calendar_step per event_number
    mean_trace <- spag_df %>%
      group_by(event_number) %>%
      summarise(mean_step = mean(calendar_step, na.rm = TRUE), .groups = "drop")

    p2 <- ggplot() +
      geom_line(
        data    = spag_df,
        aes(x = event_number, y = calendar_step, group = trace_id),
        colour  = "grey70",
        alpha   = 0.25,
        linewidth = 0.3
      ) +
      geom_line(
        data      = mean_trace,
        aes(x = event_number, y = mean_step),
        colour    = "firebrick",
        linewidth = 1.2
      ) +
      labs(
        title    = "Superposed Epoch: Individual Platform Trajectories (np=27, \u03c1=0.15, \u03b1=10)",
        subtitle = "Light grey = individual platform-iteration traces; red = mean",
        x        = "Event number (burst rank within platform-iteration)",
        y        = "Calendar step of burst"
      ) +
      theme_bw(base_size = 11)

    ggsave(
      file.path(fig_dir, "spaghetti_epoch.pdf"),
      p2,
      width = 7, height = 5, dpi = 300
    )
    message("  Saved spaghetti_epoch.pdf")
  }
}

# ===========================================================================
# Figure 3: Burst size marginal distributions (3x3 grid: alpha x n_platforms)
# ===========================================================================

message("Figure 3: burst_marginals.pdf ...")

burst_path <- file.path(dist_dir, "burst_sizes.csv")

if (!file.exists(burst_path)) {
  message("  WARNING: ", burst_path, " not found — skipping figure 3")
} else {
  bursts <- read.csv(burst_path, stringsAsFactors = FALSE)

  bursts <- bursts %>%
    mutate(
      alpha_fac      = factor(paste0("alpha==", alpha),
                               levels = paste0("alpha==", sort(unique(alpha)))),
      np_fac         = factor(paste0("N[p]==", n_platforms),
                               levels = paste0("N[p]==", sort(unique(n_platforms))))
    )

  # Check whether log scale is warranted (>2 orders of magnitude)
  brange <- range(bursts$burst_size, na.rm = TRUE)
  use_log <- (brange[2] / max(brange[1], 1)) > 100

  p3 <- ggplot(bursts, aes(x = burst_size, fill = alpha_fac)) +
    geom_histogram(bins = 30, alpha = 0.75, colour = "white") +
    facet_grid(alpha_fac ~ np_fac, labeller = label_parsed) +
    scale_fill_brewer(palette = "Set1", guide = "none") +
    labs(
      title = "Burst Size Marginal Distributions",
      x     = if (use_log) "Burst size (log scale)" else "Burst size",
      y     = "Count"
    ) +
    theme_bw(base_size = 10) +
    theme(strip.background = element_rect(fill = "grey90"))

  if (use_log) {
    p3 <- p3 + scale_x_log10(labels = label_comma())
  }

  ggsave(
    file.path(fig_dir, "burst_marginals.pdf"),
    p3,
    width = 7, height = 5, dpi = 300
  )
  message("  Saved burst_marginals.pdf")
}

# ===========================================================================
# Figure 4: Enclave settling time distribution
# ===========================================================================

message("Figure 4: enclave_settling.pdf ...")

enclave_path <- file.path(dist_dir, "enclave_metrics.csv")

if (!file.exists(enclave_path)) {
  message("  WARNING: ", enclave_path, " not found — skipping figure 4")
} else {
  enclaves <- read.csv(enclave_path, stringsAsFactors = FALSE)

  # settling_step may be NA/empty for platforms that never settled
  enclaves <- enclaves %>%
    filter(!is.na(settling_step)) %>%
    mutate(
      disruption_cat = case_when(
        n_disruptions == 0 ~ "0",
        n_disruptions == 1 ~ "1",
        TRUE               ~ "2+"
      ),
      disruption_cat = factor(disruption_cat, levels = c("0", "1", "2+")),
      alpha_label    = paste0("alpha == ", alpha)
    )

  if (nrow(enclaves) == 0) {
    message("  WARNING: no valid settling_step rows — skipping figure 4")
  } else {
    p4 <- ggplot(enclaves, aes(x = settling_step, fill = disruption_cat)) +
      geom_histogram(bins = 25, colour = "white", alpha = 0.85, position = "stack") +
      facet_wrap(~ alpha_label, labeller = label_parsed, ncol = 3) +
      scale_fill_brewer(palette = "Set2", name = "N disruptions") +
      labs(
        title    = "Enclave Settling Time Distribution",
        subtitle = "Colored by number of post-settlement disruptions",
        x        = "Settling step",
        y        = "Count (coalition platform-iterations)"
      ) +
      theme_bw(base_size = 11) +
      theme(
        strip.background = element_rect(fill = "grey90"),
        legend.position  = "right"
      )

    ggsave(
      file.path(fig_dir, "enclave_settling.pdf"),
      p4,
      width = 7, height = 5, dpi = 300
    )
    message("  Saved enclave_settling.pdf")
  }
}

message("Done. Figures written to ", fig_dir)
