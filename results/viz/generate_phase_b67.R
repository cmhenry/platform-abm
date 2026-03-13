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
# B7: Extremist Concentration Bar Chart
# ===========================================================================

message("\n--- B7: Extremist Concentration ---")

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
  message("  SKIPPING B7: no extremist count data")
} else {
  ext_data <- ext_data %>%
    mutate(
      governance = factor(governance, levels = c("Direct", "Coalition", "Algorithmic")),
      alpha_f = factor(alpha, levels = alpha_levels),
      np_f = factor(np, levels = rev(np_levels)),
      np_lab = factor(paste0("N[p] == ", np), levels = paste0("N[p] == ", np_levels)),
      rho_lab = paste0("rho[e] == ", sprintf("%.2f", rho))
    )

  p_b7 <- ggplot(ext_data, aes(x = factor(alpha_f), y = fraction, fill = governance)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65) +
    geom_hline(yintercept = 1/3, linetype = "dashed", colour = "grey50", linewidth = 0.4) +
    facet_grid(np_lab ~ rho_lab, labeller = label_parsed) +
    scale_fill_manual(name = "Governance", values = gov_colours) +
    scale_y_continuous(
      limits = c(0, 0.7),
      breaks = seq(0, 0.7, 0.1),
      labels = scales::percent_format(accuracy = 1)
    ) +
    annotate("text", x = 0.6, y = 0.35, label = "Equal share",
             size = 2.5, colour = "grey50", hjust = 0) +
    labs(
      x = expression("Parasitism intensity (" * alpha * ")"),
      y = "Fraction of extremists",
      title = "Extremist concentration by governance type"
    ) +
    theme(
      legend.position = "bottom",
      panel.grid.major.x = element_blank()
    )

  save_fig(p_b7, "fig_extremist_concentration", width = 10, height = 8)
}


message("\n=== B6-B7 figures generated ===")
