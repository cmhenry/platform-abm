#!/usr/bin/env Rscript
# Phase B: Publication-quality visualizations for platform ABM experiment 2
# Generates figures B1-B5 as PDF + PNG (300 DPI)

library(tidyverse)
library(jsonlite)
library(patchwork)
library(viridis)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

base_dir <- file.path(dirname(dirname(dirname(
  if (interactive()) rstudioapi::getActiveDocumentContext()$path
  else commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))] |>
    sub("--file=", "", x = _)
))), "results")

# Fallback for sourcing
if (!dir.exists(file.path(base_dir, "exp2"))) {
  base_dir <- "results"
}

exp2_dir <- file.path(base_dir, "exp2")
viz_dir  <- file.path(base_dir, "viz")
dir.create(viz_dir, showWarnings = FALSE, recursive = TRUE)

# Factorial levels
np_levels    <- c(3, 9, 27)
rho_levels   <- c(0.05, 0.10, 0.15)
alpha_levels <- c(2, 5, 10)

# Helper: config directory name
config_name <- function(np, rho, alpha) {
  rho_str <- sprintf("%03d", as.integer(rho * 100))
  sprintf("exp2_np%d_rho%s_alpha%d", np, rho_str, as.integer(alpha))
}

# ---------------------------------------------------------------------------
# Theme: publication-quality defaults
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

# Helper to save both PDF and PNG
save_fig <- function(plot, name, width = 10, height = 6) {
  ggsave(file.path(viz_dir, paste0(name, ".pdf")), plot,
         width = width, height = height, device = "pdf")
  ggsave(file.path(viz_dir, paste0(name, ".png")), plot,
         width = width, height = height, dpi = 300)
  message("Saved: ", name)
}

# ---------------------------------------------------------------------------
# Data loading: summary.csv across all configs
# ---------------------------------------------------------------------------

load_summary_row <- function(np, rho, alpha, measure) {
  cfg <- config_name(np, rho, alpha)
  path <- file.path(exp2_dir, cfg, "summary.csv")
  if (!file.exists(path)) return(NULL)
  df <- read.csv(path, stringsAsFactors = FALSE)
  row <- df[df$Measure == measure, , drop = FALSE]
  if (nrow(row) == 0) return(NULL)
  # Coerce numeric columns (guards against mixed-type CSVs)
  for (col in c("Mean", "SD", "CI_Lower", "CI_Upper", "Median", "Min", "Max", "N")) {
    row[[col]] <- as.numeric(row[[col]])
  }
  row$np    <- np
  row$rho   <- rho
  row$alpha <- alpha
  row
}

build_summary_table <- function(measure) {
  rows <- list()
  for (np in np_levels) {
    for (rho in rho_levels) {
      for (alpha in alpha_levels) {
        r <- load_summary_row(np, rho, alpha, measure)
        if (!is.null(r)) rows[[length(rows) + 1]] <- r
      }
    }
  }
  bind_rows(rows)
}

# ---------------------------------------------------------------------------
# Data loading: burst_aggregate.json across all configs
# ---------------------------------------------------------------------------

load_burst <- function(np, rho, alpha) {
  cfg  <- config_name(np, rho, alpha)
  path <- file.path(exp2_dir, cfg, "dynamics", "burst_aggregate.json")
  if (!file.exists(path)) return(NULL)
  # Replace NaN with null for valid JSON parsing
  txt <- readLines(path, warn = FALSE) |> paste(collapse = "\n")
  txt <- gsub("NaN", "null", txt)
  j <- fromJSON(txt)
  tibble(
    np    = np,
    rho   = rho,
    alpha = alpha,
    burst_rate         = j$burst_rate,
    median_burst_size  = j$median_burst_size,
    mean_burst_size    = j$mean_burst_size,
    escalation_mean_slope = j$escalation_mean_slope,
    escalation_sd_slope   = j$escalation_sd_slope,
    escalation_p_value    = j$escalation_ttest$p_value,
    escalation_n          = j$escalation_ttest$n
  )
}

burst_list <- list()
for (np in np_levels) {
  for (rho in rho_levels) {
    for (alpha in alpha_levels) {
      b <- load_burst(np, rho, alpha)
      if (!is.null(b)) burst_list[[length(burst_list) + 1]] <- b
    }
  }
}
burst_master <- bind_rows(burst_list)

# ===========================================================================
# B1: Interaction Heatmap — Normalized Mainstream Utility
# ===========================================================================

message("\n--- B1: Interaction Heatmap ---")

norm_util <- build_summary_table("norm_utility_mainstream")

# Ensure factor ordering for axes
norm_util <- norm_util %>%

  mutate(
    np_f    = factor(np, levels = np_levels),
    alpha_f = factor(alpha, levels = alpha_levels),
    rho_lab = paste0("rho[e] == ", sprintf("%.2f", rho))
  )

# Shared color range across panels
fill_range <- range(norm_util$Mean, na.rm = TRUE)

p_b1 <- ggplot(norm_util, aes(x = alpha_f, y = np_f, fill = Mean)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.3f", Mean)), size = 3.5, colour = "black") +
  facet_wrap(~ rho_lab, ncol = 3, labeller = label_parsed) +
  scale_fill_viridis_c(
    name = "Norm. mainstream\nutility",
    limits = fill_range,
    option = "viridis"
  ) +
  labs(
    x = expression("Parasitism intensity (" * alpha * ")"),
    y = expression("Number of platforms (" * N[p] * ")")
  ) +
  theme(
    panel.grid = element_blank(),
    axis.ticks = element_blank()
  )

save_fig(p_b1, "fig_interaction_heatmap", width = 11, height = 4.5)


# ===========================================================================
# B2: Escalation Slope Heatmap
# ===========================================================================

message("\n--- B2: Escalation Slope Heatmap ---")

esc <- burst_master %>%
  mutate(
    np_f    = factor(np, levels = np_levels),
    alpha_f = factor(alpha, levels = alpha_levels),
    rho_lab = paste0("rho[e] == ", sprintf("%.2f", rho)),
    sig_label = ifelse(escalation_p_value > 0.05, "ns", "")
  )

# Symmetric diverging range centred at 0
slope_max <- max(abs(esc$escalation_mean_slope), na.rm = TRUE)

p_b2 <- ggplot(esc, aes(x = alpha_f, y = np_f, fill = escalation_mean_slope)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(
    aes(label = ifelse(sig_label == "ns",
                        paste0(sprintf("%.2f", escalation_mean_slope), "\nns"),
                        sprintf("%.2f", escalation_mean_slope))),
    size = 3.2, colour = "black", lineheight = 0.85
  ) +
  facet_wrap(~ rho_lab, ncol = 3, labeller = label_parsed) +
  scale_fill_distiller(
    name = "Mean escalation\nslope",
    palette = "RdBu",
    direction = -1,
    limits = c(-slope_max, slope_max)
  ) +
  labs(
    x = expression("Parasitism intensity (" * alpha * ")"),
    y = expression("Number of platforms (" * N[p] * ")")
  ) +
  theme(
    panel.grid = element_blank(),
    axis.ticks = element_blank()
  )

save_fig(p_b2, "fig_escalation_heatmap", width = 11, height = 4.5)


# ===========================================================================
# B3: Governance Utility Divergence Plot
# ===========================================================================

message("\n--- B3: Governance Divergence ---")

gov_types <- c("algorithmic", "coalition", "direct")

load_gov_utility <- function(rho_val) {
  rows <- list()
  for (np in np_levels) {
    for (alpha in alpha_levels) {
      for (gov in gov_types) {
        measure <- paste0("avg_utility_mainstream_", gov)
        r <- load_summary_row(np, rho_val, alpha, measure)
        if (!is.null(r)) {
          r$governance <- gov
          r$SE <- (r$CI_Upper - r$CI_Lower) / (2 * 1.96)
          rows[[length(rows) + 1]] <- r
        }
      }
    }
  }
  bind_rows(rows)
}

# Primary: rho = 0.15; fallback rho = 0.10
gov_data_015 <- load_gov_utility(0.15)
gov_data_010 <- load_gov_utility(0.10)

# Use 0.15 as primary, note missing point
gov_data <- gov_data_015
rho_used <- 0.15

gov_data <- gov_data %>%
  mutate(
    np_lab     = paste0("N[p] == ", np),
    np_f       = factor(np_lab, levels = paste0("N[p] == ", np_levels)),
    governance = str_to_title(governance),
    gov_f      = factor(governance, levels = c("Direct", "Coalition", "Algorithmic"))
  )

# Colour palette: colorblind-safe
gov_colours <- c(
  "Direct"      = "#D55E00",
  "Coalition"   = "#009E73",
  "Algorithmic" = "#0072B2"
)

p_b3 <- ggplot(gov_data, aes(x = alpha, y = Mean, colour = gov_f, group = gov_f)) +
  geom_hline(yintercept = 5.0, linetype = "dashed", colour = "grey50", linewidth = 0.5) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE), width = 0.4, linewidth = 0.5) +
  facet_wrap(~ np_f, ncol = 3, labeller = label_parsed) +
  scale_colour_manual(name = "Governance", values = gov_colours) +
  scale_x_continuous(breaks = alpha_levels) +
  annotate("text", x = 2.3, y = 5.15, label = "Random baseline",
           size = 2.8, colour = "grey50", hjust = 0) +
  labs(
    x = expression("Parasitism intensity (" * alpha * ")"),
    y = "Mainstream utility",
    subtitle = bquote(rho[e] == .(rho_used))
  ) +
  theme(legend.position = "bottom")

save_fig(p_b3, "fig_governance_divergence", width = 10, height = 5)


# ===========================================================================
# B4: Burst Heatmap Grid
# ===========================================================================

message("\n--- B4: Burst Heatmap ---")

burst_015 <- burst_master %>%
  filter(rho == 0.15) %>%
  mutate(
    np_f    = factor(np, levels = np_levels),
    alpha_f = factor(alpha, levels = alpha_levels)
  )

# Panel 1: Median burst size
p_b4a <- ggplot(burst_015, aes(x = alpha_f, y = np_f, fill = median_burst_size)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.0f", median_burst_size)), size = 4, colour = "black") +
  scale_fill_viridis_c(name = "Median\nburst size", option = "magma", direction = -1) +
  labs(
    title = "Median Burst Size",
    x = expression(alpha),
    y = expression(N[p])
  ) +
  theme(panel.grid = element_blank(), axis.ticks = element_blank())

# Panel 2: Burst rate
p_b4b <- ggplot(burst_015, aes(x = alpha_f, y = np_f, fill = burst_rate)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.2f", burst_rate)), size = 4, colour = "black") +
  scale_fill_viridis_c(name = "Burst\nrate", option = "magma", direction = -1) +
  labs(
    title = "Burst Rate",
    x = expression(alpha),
    y = expression(N[p])
  ) +
  theme(panel.grid = element_blank(), axis.ticks = element_blank())

p_b4 <- (p_b4a | p_b4b) +
  plot_annotation(
    subtitle = expression(rho[e] == 0.15),
    theme = theme(plot.subtitle = element_text(hjust = 0.5, size = 12))
  )

save_fig(p_b4, "fig_burst_heatmap", width = 10, height = 4.5)


# ===========================================================================
# B5: Enclave Trajectory Plot
# ===========================================================================

message("\n--- B5: Enclave Trajectory ---")

enclave_path <- file.path(exp2_dir, "exp2_np27_rho015_alpha5",
                           "dynamics", "enclaves.json")
enc <- fromJSON(enclave_path)

# Build long-format data from per-platform homogeneity series
enc_long <- map_dfr(names(enc), function(pid) {
  tibble(
    platform = pid,
    step     = seq_along(enc[[pid]]$homogeneity_series),
    homogeneity = enc[[pid]]$homogeneity_series
  )
})

# Compute mean across platforms per step
enc_mean <- enc_long %>%
  group_by(step) %>%
  summarise(homogeneity = mean(homogeneity), .groups = "drop") %>%
  mutate(platform = "mean")

p_b5 <- ggplot() +
  geom_hline(yintercept = 0.9, linetype = "dashed", colour = "grey50", linewidth = 0.5) +
  geom_line(data = enc_long,
            aes(x = step, y = homogeneity, group = platform),
            colour = "grey70", linewidth = 0.4, alpha = 0.7) +
  geom_line(data = enc_mean,
            aes(x = step, y = homogeneity),
            colour = "#009E73", linewidth = 1.2) +
  annotate("text", x = 85, y = 0.92, label = "Enclave threshold (0.9)",
           size = 3, colour = "grey50") +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.2)) +
  labs(
    x = "Simulation step",
    y = "Coalition platform homogeneity",
    subtitle = expression(N[p] == 27 ~ ", " ~ rho[e] == 0.15 ~ ", " ~ alpha == 5)
  )

save_fig(p_b5, "fig_enclave_trajectory", width = 8, height = 5)


message("\n=== All Phase B figures generated ===")
