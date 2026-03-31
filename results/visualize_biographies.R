#!/usr/bin/env Rscript
# visualize_biographies.R
#
# Multi-panel platform biography figure for platform-ABM.
# Shows per-platform extremist population trajectories across the four
# representative behavioral regimes, using platform_detail.csv from
# the biography config runs.
#
# Regime panels (one per config):
#   A. Worst case    — N_p=27, alpha=10  (enclave formation + high raiding)
#   B. Oscillatory   — N_p=9,  alpha=10  (recovery cycles)
#   C. Turbulence    — N_p=3,  alpha=10  (continuous displacement)
#   D. Moderate      — N_p=27, alpha=5   (quiet sorting, resilience baseline)
#
# Iteration selection: the iteration whose total per-platform extremist
# variation (sum of within-platform SD across steps) is closest to the
# median across all 50 iterations — i.e., the "typical" run, not a
# best- or worst-case cherry-pick.
#
# Usage (from project root):
#   Rscript results/visualize_biographies.R
#
# Output: results/figures/biography_panel.pdf  (and .png)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(scales)
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

detail_base <- "results/exp2_detail/exp2"
fig_dir     <- "results/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

configs <- list(
  list(
    name   = "exp2_np27_rho015_alpha10",
    label  = "A. Worst case\n(N[p]==27, alpha==10)",
    regime = "Enclave + raiding"
  ),
  list(
    name   = "exp2_np9_rho015_alpha10",
    label  = "B. Oscillatory\n(N[p]==9, alpha==10)",
    regime = "Recovery cycles"
  ),
  list(
    name   = "exp2_np3_rho015_alpha10",
    label  = "C. Continuous turbulence\n(N[p]==3, alpha==10)",
    regime = "Continuous displacement"
  ),
  list(
    name   = "exp2_np27_rho015_alpha5",
    label  = "D. Moderate\n(N[p]==27, alpha==5)",
    regime = "Quiet sorting"
  )
)

gov_colors <- c(
  "direct"      = "#e41a1c",
  "algorithmic" = "#377eb8",
  "coalition"   = "#4daf4a"
)

# ---------------------------------------------------------------------------
# Helper: select representative iteration
# ---------------------------------------------------------------------------

select_representative_iter <- function(df) {
  # For each iteration, compute sum of per-platform SD of n_extremist across steps.
  # Pick the iteration closest to the median of that distribution.
  iter_var <- df %>%
    group_by(iteration, platform_id) %>%
    summarise(plat_sd = sd(n_extremist), .groups = "drop") %>%
    group_by(iteration) %>%
    summarise(total_var = sum(plat_sd, na.rm = TRUE), .groups = "drop")

  med_var <- median(iter_var$total_var)
  best    <- iter_var %>%
    mutate(dist = abs(total_var - med_var)) %>%
    slice_min(dist, n = 1) %>%
    pull(iteration)
  best[[1]]
}

# ---------------------------------------------------------------------------
# Build one panel per config
# ---------------------------------------------------------------------------

make_panel <- function(cfg) {
  path <- file.path(detail_base, cfg$name, "platform_detail.csv")
  if (!file.exists(path)) {
    stop("Missing: ", path)
  }
  message("  Loading ", cfg$name, " ...")

  df <- read.csv(path, stringsAsFactors = FALSE) %>%
    mutate(
      platform_id     = factor(platform_id),
      governance_type = factor(governance_type, levels = c("direct", "algorithmic", "coalition"))
    )

  rep_iter <- select_representative_iter(df)
  message("    Selected iteration: ", rep_iter)

  sub <- df %>%
    filter(iteration == rep_iter) %>%
    mutate(
      total_pop     = n_mainstream + n_extremist,
      extremist_frac = ifelse(total_pop > 0, n_extremist / total_pop, 0)
    )

  n_platforms <- n_distinct(sub$platform_id)

  # Line size and alpha: fewer platforms = bolder lines
  lsize <- if (n_platforms <= 3) 1.0 else if (n_platforms <= 9) 0.7 else 0.45
  lalpha <- if (n_platforms <= 3) 1.0 else if (n_platforms <= 9) 0.85 else 0.65

  # Governance-level mean overlay
  gov_mean <- sub %>%
    group_by(step, governance_type) %>%
    summarise(mean_frac = mean(extremist_frac), .groups = "drop")

  p <- ggplot(sub, aes(x = step, y = extremist_frac,
                       group = platform_id, colour = governance_type)) +
    geom_line(linewidth = lsize, alpha = lalpha) +
    geom_line(
      data = gov_mean,
      aes(x = step, y = mean_frac, colour = governance_type, group = governance_type),
      linewidth = 1.4, alpha = 1, linetype = "solid",
      inherit.aes = FALSE
    ) +
    scale_colour_manual(
      values = gov_colors,
      labels = c("Direct", "Algorithmic", "Coalition"),
      name   = "Governance type"
    ) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
    scale_x_continuous(breaks = c(0, 25, 50, 75, 100)) +
    labs(
      title    = cfg$label,
      subtitle = paste0("iter ", rep_iter, "  \u2022  ", n_platforms, " platforms"),
      x        = "Step",
      y        = "Extremist fraction"
    ) +
    theme_bw(base_size = 10) +
    theme(
      legend.position  = "none",
      plot.title       = element_text(size = 9, face = "bold"),
      plot.subtitle    = element_text(size = 7.5, colour = "grey40"),
      axis.title       = element_text(size = 8),
      axis.text        = element_text(size = 7),
      panel.grid.minor = element_blank()
    )

  p
}

# ---------------------------------------------------------------------------
# Assemble panels
# ---------------------------------------------------------------------------

message("Building biography panels ...")
panels <- lapply(configs, make_panel)

# Arrange 2x2, collect shared legend via patchwork
combined <- (panels[[1]] | panels[[2]]) /
            (panels[[3]] | panels[[4]])

combined <- combined +
  plot_layout(guides = "collect") +
  plot_annotation(
    title   = "Platform Biographies: Extremist Population Dynamics Across Behavioral Regimes",
    caption = paste0(
      "Each line = one platform's extremist fraction over 100 steps (representative iteration per regime).\n",
      "Bold lines = governance-type mean. Thin lines = individual platforms.\n",
      "Color: red = direct, blue = algorithmic, green = coalition."
    ),
    theme = theme(
      plot.title   = element_text(size = 11, face = "bold"),
      plot.caption = element_text(size = 7, colour = "grey50", hjust = 0)
    )
  ) &
  theme(legend.position = "bottom")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

pdf_path <- file.path(fig_dir, "biography_panel.pdf")
png_path <- file.path(fig_dir, "biography_panel.png")

message("Saving ", pdf_path, " ...")
ggsave(pdf_path, combined, width = 10, height = 8, device = "pdf")

message("Saving ", png_path, " ...")
ggsave(png_path, combined, width = 10, height = 8, dpi = 150, device = "png")

message("Done. Output: ", fig_dir, "/biography_panel.{pdf,png}")
