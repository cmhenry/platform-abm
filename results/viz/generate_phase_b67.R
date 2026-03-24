#!/usr/bin/env Rscript
# Phase B6-B7: Superposed Epoch Plot and Extremist Concentration Bar Chart
# Generates publication-quality PDF + PNG (300 DPI)

library(tidyverse)
library(jsonlite)
library(patchwork)

# ---------------------------------------------------------------------------
# Setup (mirrors generate_phase_b.R)
# ---------------------------------------------------------------------------

base_dir <- file.path(dirname(dirname(dirname(
  if (interactive()) rstudioapi::getActiveDocumentContext()$path
  else commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))] |>
    sub("--file=", "", x = _)
))), "results")

if (!dir.exists(file.path(base_dir, "exp2"))) {
  base_dir <- "results"
}

exp2_dir <- file.path(base_dir, "exp2")
viz_dir  <- file.path(base_dir, "viz")
dir.create(viz_dir, showWarnings = FALSE, recursive = TRUE)

np_levels    <- c(3, 9, 27)
rho_levels   <- c(0.05, 0.10, 0.15)
alpha_levels <- c(2, 5, 10)

config_name <- function(np, rho, alpha) {
  rho_str <- sprintf("%03d", as.integer(rho * 100))
  sprintf("exp2_np%d_rho%s_alpha%d", np, rho_str, as.integer(alpha))
}

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

theme_pub <- theme_minimal(base_size = 12) +
  theme(
    panel.background  = element_rect(fill = "white", colour = NA),
    plot.background   = element_rect(fill = "white", colour = NA),
    panel.grid.major  = element_line(colour = "grey90", linewidth = 0.3),
    panel.grid.minor  = element_blank(),
    strip.text        = element_text(size = 12, face = "bold"),
    axis.title        = element_text(size = 12),
    axis.text         = element_text(size = 10),
    legend.text       = element_text(size = 10),
    legend.title      = element_text(size = 11),
    plot.title        = element_text(size = 13, face = "bold", hjust = 0.5),
    plot.margin       = margin(8, 8, 8, 8)
  )

theme_set(theme_pub)

save_fig <- function(plot, name, width = 10, height = 6) {
  ggsave(file.path(viz_dir, paste0(name, ".pdf")), plot,
         width = width, height = height, device = "pdf")
  ggsave(file.path(viz_dir, paste0(name, ".png")), plot,
         width = width, height = height, dpi = 300)
  message("Saved: ", name)
}

# Colorblind-safe governance palette
gov_colours <- c(
  "Direct"      = "#D55E00",
  "Coalition"   = "#009E73",
  "Algorithmic" = "#0072B2"
)

# ===========================================================================
# B6: Superposed Epoch Plot (baseline-normalized)
# ===========================================================================

message("\n--- B6: Superposed Epoch (baseline-normalized) ---")

# Read pre-computed baselined JSON files (from recompute_epoch.py)
epoch_files <- list(
  list(file = "epoch_baselined_exp2_np3_rho015_alpha10.json",
       label = expression(N[p] == 3 ~ ", " ~ alpha == 10)),
  list(file = "epoch_baselined_exp2_np9_rho015_alpha10.json",
       label = expression(N[p] == 9 ~ ", " ~ alpha == 10)),
  list(file = "epoch_baselined_exp2_np27_rho015_alpha5.json",
       label = expression(N[p] == 27 ~ ", " ~ alpha == 5))
)

# Config display strings (used for faceting)
config_strings <- c(
  "epoch_baselined_exp2_np3_rho015_alpha10.json"  = "N[p]==3~\",\"~alpha==10",
  "epoch_baselined_exp2_np9_rho015_alpha10.json"  = "N[p]==9~\",\"~alpha==10",
  "epoch_baselined_exp2_np27_rho015_alpha5.json"   = "N[p]==27~\",\"~alpha==5"
)

count_rows <- list()
util_rows  <- list()

for (ef in epoch_files) {
  path <- file.path(viz_dir, ef$file)
  if (!file.exists(path)) {
    message("  Missing: ", path)
    next
  }
  j <- fromJSON(path)
  steps <- j$relative_steps
  cfg_str <- config_strings[[ef$file]]

  # Count series (3 governance types)
  for (gov_info in list(
    list(prefix = "direct_count",      gov = "Direct"),
    list(prefix = "coalition_count",   gov = "Coalition"),
    list(prefix = "algorithmic_count", gov = "Algorithmic")
  )) {
    count_rows[[length(count_rows) + 1]] <- tibble(
      step = steps,
      delta_mean = j[[paste0(gov_info$prefix, "_delta_mean")]],
      delta_se   = j[[paste0(gov_info$prefix, "_delta_se")]],
      governance = gov_info$gov,
      config     = cfg_str
    )
  }

  # Mainstream utility (single line per config)
  util_rows[[length(util_rows) + 1]] <- tibble(
    step       = steps,
    delta_mean = j$mainstream_util_delta_mean,
    delta_se   = j$mainstream_util_delta_se,
    governance = "Mainstream",
    config     = cfg_str
  )
}

count_all <- bind_rows(count_rows)
util_all  <- bind_rows(util_rows)

if (nrow(count_all) == 0) {
  message("  SKIPPING B6: no baselined epoch data found")
} else {
  # Order configs top → bottom
  config_order <- unname(config_strings)
  count_all$config <- factor(count_all$config, levels = config_order)
  util_all$config  <- factor(util_all$config,  levels = config_order)

  count_all$governance <- factor(count_all$governance,
                                  levels = c("Direct", "Coalition", "Algorithmic"))

  mainstream_colour <- c("Mainstream" = "#333333")

  # --- Left panels: Δ community count ---
  p_count <- ggplot(count_all,
                    aes(x = step, y = delta_mean, colour = governance, fill = governance)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.4) +
    geom_ribbon(aes(ymin = delta_mean - delta_se, ymax = delta_mean + delta_se),
                alpha = 0.15, colour = NA) +
    geom_line(linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40", linewidth = 0.4) +
    facet_wrap(~ config, ncol = 1, scales = "free_y", labeller = label_parsed) +
    scale_colour_manual(name = "Governance", values = gov_colours) +
    scale_fill_manual(name = "Governance", values = gov_colours) +
    scale_x_continuous(breaks = seq(-8, 8, 2)) +
    labs(x = "Relative step (t = 0 at raid)", y = expression(Delta ~ "community count")) +
    theme(legend.position = "bottom")

  # --- Right panels: Δ mainstream utility ---
  p_util <- ggplot(util_all,
                   aes(x = step, y = delta_mean, colour = governance, fill = governance)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.4) +
    geom_ribbon(aes(ymin = delta_mean - delta_se, ymax = delta_mean + delta_se),
                alpha = 0.15, colour = NA) +
    geom_line(linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40", linewidth = 0.4) +
    facet_wrap(~ config, ncol = 1, scales = "free_y", labeller = label_parsed) +
    scale_colour_manual(name = NULL, values = mainstream_colour) +
    scale_fill_manual(name = NULL, values = mainstream_colour) +
    scale_x_continuous(breaks = seq(-8, 8, 2)) +
    labs(x = "Relative step (t = 0 at raid)", y = expression(Delta ~ "mainstream utility")) +
    theme(legend.position = "bottom")

  # Combine: 2 columns side by side
  p_b6 <- (p_count | p_util) +
    plot_layout(widths = c(1, 1)) +
    plot_annotation(
      title = "Superposed epoch analysis: baseline-normalized displacement",
      subtitle = expression(rho[e] == 0.15 ~ " -- baseline: mean of t = -8 to -4; shaded bands: +/-1 SE"),
      theme = theme(
        plot.title    = element_text(size = 13, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 11)
      )
    )

  save_fig(p_b6, "fig_superposed_epoch", width = 10, height = 10)
}


# ===========================================================================
# B7a: Simplified Extremist Concentration (3-panel, averaged over alpha)
# ===========================================================================

message("\n--- B7a: Simplified Extremist Concentration ---")

load_summary_row <- function(np, rho, alpha, measure) {
  cfg <- config_name(np, rho, alpha)
  path <- file.path(exp2_dir, cfg, "summary.csv")
  if (!file.exists(path)) return(NULL)
  df <- read.csv(path, stringsAsFactors = FALSE)
  row <- df[df$Measure == measure, , drop = FALSE]
  if (nrow(row) == 0) return(NULL)
  for (col in c("Mean", "SD", "CI_Lower", "CI_Upper", "Median", "Min", "Max", "N")) {
    row[[col]] <- as.numeric(row[[col]])
  }
  row$np    <- np
  row$rho   <- rho
  row$alpha <- alpha
  row
}

# Collect extremist counts by governance type
ext_rows <- list()
for (np in np_levels) {
  for (rho in rho_levels) {
    for (alpha in alpha_levels) {
      direct <- load_summary_row(np, rho, alpha, "final_count_extremist_direct")
      coalition <- load_summary_row(np, rho, alpha, "final_count_extremist_coalition")
      algo <- load_summary_row(np, rho, alpha, "final_count_extremist_algorithmic")

      if (is.null(direct) || is.null(coalition) || is.null(algo)) next

      total <- direct$Mean + coalition$Mean + algo$Mean
      if (is.na(total) || total == 0) next

      ext_rows[[length(ext_rows) + 1]] <- tibble(
        np = np, rho = rho, alpha = alpha,
        governance = "Direct", fraction = direct$Mean / total
      )
      ext_rows[[length(ext_rows) + 1]] <- tibble(
        np = np, rho = rho, alpha = alpha,
        governance = "Coalition", fraction = coalition$Mean / total
      )
      ext_rows[[length(ext_rows) + 1]] <- tibble(
        np = np, rho = rho, alpha = alpha,
        governance = "Algorithmic", fraction = algo$Mean / total
      )
    }
  }
}

ext_data <- bind_rows(ext_rows)

if (nrow(ext_data) == 0) {
  message("  SKIPPING B7a: no extremist count data")
} else {
  # Average fraction across alpha values, track min/max for error bars
  ext_summary <- ext_data %>%
    group_by(np, rho, governance) %>%
    summarise(
      frac_mean = mean(fraction),
      frac_min  = min(fraction),
      frac_max  = max(fraction),
      .groups = "drop"
    ) %>%
    mutate(
      governance = factor(governance, levels = c("Direct", "Coalition", "Algorithmic")),
      np_f = factor(np, levels = np_levels),
      rho_lab = paste0("rho[e] == ", sprintf("%.2f", rho))
    )

  p_b7a <- ggplot(ext_summary, aes(x = np_f, y = frac_mean, fill = governance)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65) +
    geom_errorbar(aes(ymin = frac_min, ymax = frac_max),
                  position = position_dodge(width = 0.75), width = 0.25, linewidth = 0.4) +
    geom_hline(yintercept = 1/3, linetype = "dashed", colour = "grey50", linewidth = 0.4) +
    facet_wrap(~ rho_lab, ncol = 3, labeller = label_parsed) +
    scale_fill_manual(name = "Governance", values = gov_colours) +
    scale_y_continuous(
      limits = c(0, 0.6),
      breaks = seq(0, 0.6, 0.1),
      labels = scales::percent_format(accuracy = 1)
    ) +
    labs(
      x = expression(N[p]),
      y = "Fraction of extremists",
      title = expression("Extremist concentration by governance type (averaged over " * alpha * ")")
    ) +
    theme(
      legend.position = "bottom",
      panel.grid.major.x = element_blank()
    )

  save_fig(p_b7a, "fig_extremist_concentration_simple", width = 10, height = 4)
}


# ===========================================================================
# B7b: Overrepresentation Ratio Heatmap (rho = 0.15 only)
# ===========================================================================

message("\n--- B7b: Overrepresentation Ratio Heatmap ---")

# Collect both extremist and total community counts
ratio_rows <- list()
for (np in np_levels) {
  for (alpha in alpha_levels) {
    rho <- 0.15

    # Extremist counts
    ext_d <- load_summary_row(np, rho, alpha, "final_count_extremist_direct")
    ext_c <- load_summary_row(np, rho, alpha, "final_count_extremist_coalition")
    ext_a <- load_summary_row(np, rho, alpha, "final_count_extremist_algorithmic")

    # Total community counts
    tot_d <- load_summary_row(np, rho, alpha, "final_count_direct")
    tot_c <- load_summary_row(np, rho, alpha, "final_count_coalition")
    tot_a <- load_summary_row(np, rho, alpha, "final_count_algorithmic")

    if (is.null(ext_d) || is.null(ext_c) || is.null(ext_a)) next
    if (is.null(tot_d) || is.null(tot_c) || is.null(tot_a)) next

    total_ext <- ext_d$Mean + ext_c$Mean + ext_a$Mean
    total_pop <- tot_d$Mean + tot_c$Mean + tot_a$Mean

    if (is.na(total_ext) || total_ext == 0 || is.na(total_pop) || total_pop == 0) next

    for (info in list(
      list(gov = "Direct",      ext = ext_d$Mean, pop = tot_d$Mean),
      list(gov = "Coalition",   ext = ext_c$Mean, pop = tot_c$Mean),
      list(gov = "Algorithmic", ext = ext_a$Mean, pop = tot_a$Mean)
    )) {
      ext_frac <- info$ext / total_ext
      pop_frac <- info$pop / total_pop
      ratio <- if (pop_frac > 0) ext_frac / pop_frac else NA_real_

      ratio_rows[[length(ratio_rows) + 1]] <- tibble(
        np = np, alpha = alpha,
        governance = info$gov,
        ratio = ratio
      )
    }
  }
}

ratio_data <- bind_rows(ratio_rows)

if (nrow(ratio_data) == 0) {
  message("  SKIPPING B7b: no data for overrepresentation ratio")
} else {
  # Average ratio across alpha values
  ratio_summary <- ratio_data %>%
    group_by(np, governance) %>%
    summarise(ratio_mean = mean(ratio, na.rm = TRUE), .groups = "drop") %>%
    mutate(
      governance = factor(governance, levels = c("Direct", "Coalition", "Algorithmic")),
      np_f = factor(np, levels = np_levels),
      label = sprintf("%.1f\u00d7", ratio_mean)
    )

  p_b7b <- ggplot(ratio_summary, aes(x = np_f, y = governance, fill = ratio_mean)) +
    geom_tile(colour = "white", linewidth = 0.8) +
    geom_text(aes(label = label), size = 5) +
    scale_fill_gradient2(
      low = "#0072B2", mid = "white", high = "#D55E00", midpoint = 1,
      name = "Overrepresentation\nratio"
    ) +
    labs(
      x = expression(N[p]),
      y = NULL,
      title = "Extremist overrepresentation ratio",
      subtitle = expression(rho[e] == 0.15 ~ ", averaged over " ~ alpha)
    ) +
    theme(
      panel.grid = element_blank(),
      axis.ticks = element_blank(),
      plot.subtitle = element_text(hjust = 0.5, size = 11)
    )

  save_fig(p_b7b, "fig_overrepresentation_ratio", width = 7, height = 4)
}


message("\n=== B6-B7 figures generated ===")
