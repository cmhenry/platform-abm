#!/usr/bin/env Rscript
# visualize_exp2b.R
#
# Figures for Experiment 2b: mixed ideologue/griefer populations.
# Varies griefer fraction f_g ∈ {0, 0.25, 0.50, 0.75, 1.0} at
# fixed N_p=9, rho_e=0.10, with exp2 alpha=2 / alpha=10 configs
# as the pure-type endpoints.
#
# Produces two figures:
#   1. exp2b_governance.pdf  — mainstream utility by governance type across f_g
#   2. exp2b_dynamics.pdf    — burst and enclave dynamics across f_g
#
# Usage (from project root):
#   Rscript results/visualize_exp2b.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(scales)
  library(jsonlite)
})

fig_dir <- "results/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Governance type colors — consistent with biography_panel.R
# ---------------------------------------------------------------------------

gov_colors <- c(
  "Direct"       = "#e41a1c",
  "Coalition"    = "#4daf4a",
  "Algorithmic"  = "#377eb8"
)

# ===========================================================================
# Data assembly: pull exp2 baselines (f_g = 0, f_g = 1) and exp2b midpoints
# ===========================================================================

message("Assembling governance utility data ...")

# --- exp2 baselines at N_p=9, rho=0.10 ---
exp2_summary <- read.csv("results/exp2/summary.csv", stringsAsFactors = FALSE)

base_ideologue <- exp2_summary %>%
  filter(config_name == "exp2_np9_rho010_alpha2") %>%
  select(avg_utility_mainstream_direct,
         avg_utility_mainstream_coalition,
         avg_utility_mainstream_algorithmic,
         burst_rate, burst_size_median, escalation_mean_slope, escalation_p_value,
         enclave_mean_homogeneity, enclave_settling_step)

base_griefer <- exp2_summary %>%
  filter(config_name == "exp2_np9_rho010_alpha10") %>%
  select(avg_utility_mainstream_direct,
         avg_utility_mainstream_coalition,
         avg_utility_mainstream_algorithmic,
         burst_rate, burst_size_median, escalation_mean_slope, escalation_p_value,
         enclave_mean_homogeneity, enclave_settling_step)

# --- exp2b intermediate points ---
exp2b_summary <- read.csv("results/exp2b/exp2b_master_summary.csv",
                          stringsAsFactors = FALSE)

# Read per-config burst and enclave aggregates
read_aggregate <- function(config_dir, filename) {
  path <- file.path("results/exp2b", config_dir, filename)
  if (!file.exists(path)) return(NULL)
  fromJSON(path)
}

burst_fg025 <- read_aggregate("exp2b_fg025", "burst_aggregate.json")
burst_fg050 <- read_aggregate("exp2b_fg050", "burst_aggregate.json")
burst_fg075 <- read_aggregate("exp2b_fg075", "burst_aggregate.json")

enclave_fg025 <- read_aggregate("exp2b_fg025", "enclave_aggregate.json")
enclave_fg050 <- read_aggregate("exp2b_fg050", "enclave_aggregate.json")
enclave_fg075 <- read_aggregate("exp2b_fg075", "enclave_aggregate.json")

# ---------------------------------------------------------------------------
# Governance utility data frame (5-point gradient)
# ---------------------------------------------------------------------------

gov_df <- bind_rows(
  tibble(
    f_g       = 0.00,
    source    = "exp2 baseline",
    Direct      = base_ideologue$avg_utility_mainstream_direct,
    Coalition   = base_ideologue$avg_utility_mainstream_coalition,
    Algorithmic = base_ideologue$avg_utility_mainstream_algorithmic
  ),
  exp2b_summary %>%
    mutate(
      f_g = case_when(
        config_name == "exp2b_fg025" ~ 0.25,
        config_name == "exp2b_fg050" ~ 0.50,
        config_name == "exp2b_fg075" ~ 0.75
      ),
      source = "exp2b"
    ) %>%
    select(f_g, source,
           Direct      = avg_utility_mainstream_direct,
           Coalition   = avg_utility_mainstream_coalition,
           Algorithmic = avg_utility_mainstream_algorithmic),
  tibble(
    f_g       = 1.00,
    source    = "exp2 baseline",
    Direct      = base_griefer$avg_utility_mainstream_direct,
    Coalition   = base_griefer$avg_utility_mainstream_coalition,
    Algorithmic = base_griefer$avg_utility_mainstream_algorithmic
  )
) %>%
  pivot_longer(cols = c(Direct, Coalition, Algorithmic),
               names_to = "governance", values_to = "mainstream_utility") %>%
  mutate(governance = factor(governance, levels = c("Direct", "Coalition", "Algorithmic")))

# Add welfare loss relative to pure-ideologue baseline (f_g = 0)
baseline_vals <- gov_df %>%
  filter(f_g == 0) %>%
  select(governance, baseline = mainstream_utility)

gov_df <- gov_df %>%
  left_join(baseline_vals, by = "governance") %>%
  mutate(welfare_loss = mainstream_utility - baseline)

# ---------------------------------------------------------------------------
# Dynamics data frame (3 exp2b points only; endpoints noted as context)
# ---------------------------------------------------------------------------

dynamics_df <- tibble(
  f_g = c(0.25, 0.50, 0.75),
  burst_size_median = c(
    burst_fg025$burst_size_median,
    burst_fg050$burst_size_median,
    burst_fg075$burst_size_median
  ),
  burst_rate = c(
    burst_fg025$burst_rate,
    burst_fg050$burst_rate,
    burst_fg075$burst_rate
  ),
  escalation_slope = c(
    burst_fg025$escalation_mean_slope,
    burst_fg050$escalation_mean_slope,
    burst_fg075$escalation_mean_slope
  ),
  escalation_sig = c(
    burst_fg025$escalation_p_value < 0.05,
    burst_fg050$escalation_p_value < 0.05,
    burst_fg075$escalation_p_value < 0.05
  ),
  enclave_homogeneity = c(
    enclave_fg025$mean_homogeneity,
    enclave_fg050$mean_homogeneity,
    enclave_fg075$mean_homogeneity
  ),
  enclave_settling = c(
    enclave_fg025$settling_step_mean,
    enclave_fg050$settling_step_mean,
    enclave_fg075$settling_step_mean
  ),
  enclave_settling_sd = c(
    enclave_fg025$settling_step_sd,
    enclave_fg050$settling_step_sd,
    enclave_fg075$settling_step_sd
  )
)

# ===========================================================================
# Figure 1: exp2b_governance.pdf
#
# Left panel: mainstream utility by governance type across f_g gradient
# Right panel: welfare loss (relative to f_g=0) by governance type
#
# Both panels share x-axis (f_g). Endpoint pairs from exp2 baselines shown
# as filled points; exp2b intermediate points shown with open diamond shape.
# ===========================================================================

message("Figure 1: exp2b_governance.pdf ...")

gov_df <- gov_df %>%
  mutate(point_shape = if_else(source == "exp2 baseline", 19L, 23L))

p_util <- ggplot(gov_df, aes(x = f_g, y = mainstream_utility,
                              colour = governance, fill = governance)) +
  geom_line(linewidth = 0.8, show.legend = FALSE) +
  geom_point(aes(shape = source), size = 3, stroke = 0.5) +
  scale_colour_manual(values = gov_colors, name = "Governance") +
  scale_fill_manual(values = gov_colors, name = "Governance") +
  scale_shape_manual(values = c("exp2 baseline" = 19, "exp2b" = 23),
                     name = NULL,
                     labels = c("exp2 baseline (pure type)", "exp2b (mixed)")) +
  scale_x_continuous(
    breaks = c(0, 0.25, 0.50, 0.75, 1.0),
    labels = c("0\n(pure ideologue)", "0.25", "0.50", "0.75", "1.0\n(pure griefer)")
  ) +
  labs(
    x     = expression(f[g] ~ "(griefer fraction of extremist population)"),
    y     = "Mainstream utility",
    title = "Mainstream welfare by governance type across griefer fraction"
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position  = "bottom",
    legend.box       = "vertical",
    legend.spacing.y = unit(0.1, "cm"),
    panel.grid.minor = element_blank()
  )

p_loss <- ggplot(gov_df, aes(x = f_g, y = welfare_loss,
                              colour = governance, fill = governance)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.4) +
  geom_line(linewidth = 0.8, show.legend = FALSE) +
  geom_point(aes(shape = source), size = 3, stroke = 0.5) +
  scale_colour_manual(values = gov_colors, name = "Governance") +
  scale_fill_manual(values = gov_colors, name = "Governance") +
  scale_shape_manual(values = c("exp2 baseline" = 19, "exp2b" = 23),
                     name = NULL,
                     labels = c("exp2 baseline (pure type)", "exp2b (mixed)")) +
  scale_x_continuous(
    breaks = c(0, 0.25, 0.50, 0.75, 1.0),
    labels = c("0", "0.25", "0.50", "0.75", "1.0")
  ) +
  labs(
    x     = expression(f[g] ~ "(griefer fraction)"),
    y     = "Welfare loss relative to pure-ideologue baseline",
    title = "Differential sensitivity by governance type"
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position  = "bottom",
    legend.box       = "vertical",
    legend.spacing.y = unit(0.1, "cm"),
    panel.grid.minor = element_blank()
  )

p_gov <- p_util + p_loss +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave(
  file.path(fig_dir, "exp2b_governance.pdf"),
  p_gov,
  width = 9, height = 5, dpi = 300
)
message("  Saved exp2b_governance.pdf")

# ===========================================================================
# Figure 2: exp2b_dynamics.pdf
#
# Left panel: burst size (median) across f_g, points sized by escalation slope,
#             asterisk annotation for significant escalation
# Right panel: coalition enclave settling time (mean ± 1 SD) and homogeneity
#              (secondary axis) across f_g
#
# Only exp2b intermediate points (f_g = 0.25, 0.50, 0.75) — pure-type
# endpoints not available in this directory.
# ===========================================================================

message("Figure 2: exp2b_dynamics.pdf ...")

# Burst panel
p_burst <- ggplot(dynamics_df, aes(x = f_g)) +
  geom_col(aes(y = burst_size_median), fill = "#e41a1c", alpha = 0.75, width = 0.15) +
  geom_text(
    data = dynamics_df %>% filter(escalation_sig),
    aes(y = burst_size_median + 1.5, label = "*"),
    colour = "black", size = 5
  ) +
  geom_line(aes(y = escalation_slope * 20), colour = "#377eb8",
            linewidth = 1.0, linetype = "solid") +
  geom_point(aes(y = escalation_slope * 20), colour = "#377eb8",
             size = 3, shape = 17) +
  scale_y_continuous(
    name      = "Burst size (median, red bars)",
    limits    = c(0, 40),
    sec.axis  = sec_axis(~ . / 20,
                         name = "Escalation slope (blue line)",
                         breaks = c(0, 0.5, 1.0, 1.5))
  ) +
  scale_x_continuous(
    breaks = c(0.25, 0.50, 0.75),
    labels = c("0.25", "0.50", "0.75")
  ) +
  labs(
    x     = expression(f[g] ~ "(griefer fraction)"),
    title = "Burst dynamics across griefer fraction",
    subtitle = "* = escalation slope significant (p < 0.05)"
  ) +
  theme_bw(base_size = 11) +
  theme(
    axis.title.y.left  = element_text(colour = "#e41a1c"),
    axis.title.y.right = element_text(colour = "#377eb8"),
    panel.grid.minor   = element_blank()
  )

# Enclave panel — settling time with ± 1 SD ribbon, homogeneity on secondary axis
enclave_scaled <- dynamics_df %>%
  mutate(
    # Homogeneity ranges ~0.91-0.94; map to settling-time scale for secondary axis
    # settling range ~ 20-40; homogeneity mapped to [20, 40] range
    homogeneity_scaled = (enclave_homogeneity - 0.90) / (0.95 - 0.90) * 20 + 20
  )

p_enclave <- ggplot(enclave_scaled, aes(x = f_g)) +
  geom_ribbon(aes(ymin = enclave_settling - enclave_settling_sd,
                  ymax = enclave_settling + enclave_settling_sd),
              fill = "#4daf4a", alpha = 0.20) +
  geom_line(aes(y = enclave_settling), colour = "#4daf4a",
            linewidth = 1.0) +
  geom_point(aes(y = enclave_settling), colour = "#4daf4a",
             size = 3) +
  geom_line(aes(y = homogeneity_scaled), colour = "grey30",
            linewidth = 0.8, linetype = "dashed") +
  geom_point(aes(y = homogeneity_scaled), colour = "grey30",
             size = 3, shape = 15) +
  scale_y_continuous(
    name     = "Enclave settling step (mean ± 1 SD, green)",
    limits   = c(0, 60),
    sec.axis = sec_axis(
      ~ (. - 20) / 20 * (0.95 - 0.90) + 0.90,
      name   = "Enclave homogeneity (dashed, grey)",
      labels = label_number(accuracy = 0.01)
    )
  ) +
  scale_x_continuous(
    breaks = c(0.25, 0.50, 0.75),
    labels = c("0.25", "0.50", "0.75")
  ) +
  labs(
    x     = expression(f[g] ~ "(griefer fraction)"),
    title = "Coalition enclave formation across griefer fraction",
    subtitle = "Settling time rises; homogeneity declines as griefer fraction increases"
  ) +
  theme_bw(base_size = 11) +
  theme(
    axis.title.y.left  = element_text(colour = "#4daf4a"),
    axis.title.y.right = element_text(colour = "grey30"),
    panel.grid.minor   = element_blank()
  )

p_dyn <- p_burst + p_enclave

ggsave(
  file.path(fig_dir, "exp2b_dynamics.pdf"),
  p_dyn,
  width = 9, height = 5, dpi = 300
)
message("  Saved exp2b_dynamics.pdf")

message("Done. Figures written to ", fig_dir)
