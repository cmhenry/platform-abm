#!/usr/bin/env Rscript
# Platform biography: raiding cycle on a direct platform
# Generates fig_platform_biography.{pdf,png}

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

# ---------------------------------------------------------------------------
# Selection logic: find a representative raiding_stable platform
# ---------------------------------------------------------------------------

config <- "exp2_np9_rho015_alpha10"
config_dir <- file.path(exp2_dir, config)

message("\n--- Platform biography: selecting candidate ---")

# Load per_iter_burst_analysis.json
burst_path <- file.path(config_dir, "per_iter_burst_analysis.json")
txt <- readLines(burst_path, warn = FALSE) |> paste(collapse = "\n")
txt <- gsub("NaN", "null", txt)
burst_data <- fromJSON(txt)

# Iterate through all iterations and platforms to find candidates
candidates <- list()
for (iter_str in names(burst_data)) {
  iter_entry <- burst_data[[iter_str]]
  for (plat_id in names(iter_entry)) {
    entry <- iter_entry[[plat_id]]
    classification <- entry$classification
    n_bursts <- entry$n_bursts
    if (!is.null(classification) && classification == "raiding_stable" &&
        !is.null(n_bursts) && n_bursts >= 4 && n_bursts <= 8) {
      esc_slope <- entry$escalation_slope
      if (is.null(esc_slope)) esc_slope <- NA_real_
      candidates[[length(candidates) + 1]] <- list(
        iter = iter_str,
        plat = plat_id,
        classification = classification,
        n_bursts = n_bursts,
        escalation_slope = esc_slope,
        burst_steps = entry$burst_steps,
        burst_sizes = entry$burst_sizes
      )
    }
  }
}

message("  Found ", length(candidates), " candidates")
stopifnot(length(candidates) > 0)

# Prefer entries with escalation_slope > 0
has_positive <- sapply(candidates, function(c) {
  !is.na(c$escalation_slope) && c$escalation_slope > 0
})
if (any(has_positive)) {
  candidates <- candidates[has_positive]
  message("  Filtered to ", length(candidates), " with positive escalation slope")
}

# Pick median candidate by n_bursts
n_bursts_vec <- sapply(candidates, function(c) c$n_bursts)
ord <- order(n_bursts_vec)
candidates <- candidates[ord]
mid_idx <- ceiling(length(candidates) / 2)
selected <- candidates[[mid_idx]]

sel_iter <- selected$iter
sel_plat <- selected$plat
sel_class <- selected$classification
sel_slope <- selected$escalation_slope
sel_burst_steps <- selected$burst_steps   # 0-indexed
sel_burst_sizes <- selected$burst_sizes

message("  Selected: iteration=", sel_iter, " platform=", sel_plat,
        " classification=", sel_class, " n_bursts=", selected$n_bursts,
        " escalation_slope=", sprintf("%.2f", sel_slope))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Outflow series from dynamics/per_iter_raiding.json
raiding_path <- file.path(config_dir, "dynamics", "per_iter_raiding.json")
txt_r <- readLines(raiding_path, warn = FALSE) |> paste(collapse = "\n")
txt_r <- gsub("NaN", "null", txt_r)
raiding_data <- fromJSON(txt_r)
outflow <- raiding_data[[sel_iter]][[sel_plat]]
outflow[sapply(outflow, is.null)] <- 0
outflow <- as.numeric(outflow)

# Build outflow data frame (index i -> step i+1)
df_outflow <- tibble(
  step = seq_along(outflow),
  outflow = outflow
)

# Burst markers (convert 0-indexed to 1-indexed)
df_bursts <- tibble(
  step = as.numeric(sel_burst_steps) + 1,
  burst_size = as.numeric(sel_burst_sizes)
)

# System governance counts from step_metrics.json
metrics_path <- file.path(config_dir, "step_metrics.json")
txt_m <- readLines(metrics_path, warn = FALSE) |> paste(collapse = "\n")
txt_m <- gsub("NaN", "null", txt_m)
step_metrics <- fromJSON(txt_m)

iter_df <- step_metrics[[sel_iter]]  # data.frame with 100 rows

df_counts <- tibble(
  step        = iter_df$step,
  Direct      = iter_df$per_governance_community_count$direct,
  Coalition   = iter_df$per_governance_community_count$coalition,
  Algorithmic = iter_df$per_governance_community_count$algorithmic
)

df_counts_long <- df_counts %>%
  pivot_longer(cols = c(Direct, Coalition, Algorithmic),
               names_to = "governance", values_to = "count") %>%
  mutate(governance = factor(governance, levels = c("Direct", "Coalition", "Algorithmic")))

# ---------------------------------------------------------------------------
# Figure: 2 vertically stacked panels
# ---------------------------------------------------------------------------

message("\n--- Building biography figure ---")

# Top panel: outflow series + burst markers
p_top <- ggplot(df_outflow, aes(x = step, y = outflow)) +
  geom_area(fill = "grey80", colour = "grey50", linewidth = 0.4) +
  geom_segment(data = df_bursts,
               aes(x = step, xend = step, y = 0, yend = burst_size),
               colour = "red", linewidth = 0.9) +
  geom_text(data = df_bursts,
            aes(x = step, y = burst_size, label = burst_size),
            colour = "red", size = 3.2, nudge_y = max(abs(outflow)) * 0.05,
            fontface = "bold") +
  scale_x_continuous(breaks = seq(0, 100, 10)) +
  labs(x = "Simulation step", y = "Net community outflow")

# Bottom panel: system governance counts
p_bottom <- ggplot(df_counts_long, aes(x = step, y = count, colour = governance)) +
  geom_line(linewidth = 0.8) +
  geom_vline(data = df_bursts, aes(xintercept = step),
             linetype = "dashed", colour = "red", alpha = 0.4, linewidth = 0.4) +
  scale_colour_manual(name = "Governance", values = gov_colours) +
  scale_x_continuous(breaks = seq(0, 100, 10)) +
  labs(x = "Simulation step", y = "Community count") +
  theme(legend.position = "bottom")

# Combine with patchwork
slope_str <- ifelse(is.na(sel_slope), "NA", sprintf("%.2f", sel_slope))

p_combined <- (p_top / p_bottom) +
  plot_layout(heights = c(1, 1)) +
  plot_annotation(
    title = "Platform biography: raiding cycle on a direct platform",
    subtitle = paste0("Config: ", config, "  |  Iteration: ", sel_iter,
                      "  |  Platform: ", sel_plat,
                      "  |  Class: ", sel_class,
                      "  |  Escalation slope: ", slope_str),
    theme = theme(
      plot.title    = element_text(size = 13, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5)
    )
  )

save_fig(p_combined, "fig_platform_biography", width = 10, height = 7)

message("\n=== Platform biography figure generated ===")
