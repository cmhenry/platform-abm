#!/usr/bin/env Rscript
# Governance Flow Alluvial Diagram
# Figure 1: Stacked area chart of governance community counts over time
# Figure 2: Alluvial diagram via ggforce::geom_parallel_sets
# Generates publication-quality PDF + PNG (300 DPI)

library(tidyverse)
library(jsonlite)
library(patchwork)
library(ggforce)

# ---------------------------------------------------------------------------
# Setup (mirrors generate_phase_b67.R)
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
# Theme + helpers (from generate_phase_b67.R)
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

gov_colours <- c(
  "Direct"      = "#D55E00",
  "Coalition"   = "#009E73",
  "Algorithmic" = "#0072B2"
)

gov_levels <- c("Direct", "Coalition", "Algorithmic")

# ---------------------------------------------------------------------------
# Config list (3 entries, same as B6 epoch)
# ---------------------------------------------------------------------------

configs <- list(
  list(dir = "exp2_np3_rho015_alpha10",
       label = expression(N[p] == 3 ~ ", " ~ alpha == 10),
       label_str = "N[p]==3~\",\"~alpha==10"),
  list(dir = "exp2_np9_rho015_alpha10",
       label = expression(N[p] == 9 ~ ", " ~ alpha == 10),
       label_str = "N[p]==9~\",\"~alpha==10"),
  list(dir = "exp2_np27_rho015_alpha5",
       label = expression(N[p] == 27 ~ ", " ~ alpha == 5),
       label_str = "N[p]==27~\",\"~alpha==5")
)

# ---------------------------------------------------------------------------
# Data loading function
# ---------------------------------------------------------------------------

load_gov_trajectory <- function(config_dir_name) {
  path <- file.path(exp2_dir, config_dir_name, "step_metrics.json")
  if (!file.exists(path)) {
    message("  Missing: ", path)
    return(NULL)
  }

  message("  Loading: ", path)
  txt <- readLines(path, warn = FALSE) |> paste(collapse = "\n")
  txt <- gsub("NaN", "null", txt)
  j <- fromJSON(txt)

  n_iters <- length(j)
  # Determine number of steps from first iteration
  n_steps <- nrow(j[[1]])

  # Accumulate count matrices: n_steps x 3 (direct, coalition, algorithmic)
  count_matrix <- array(0, dim = c(n_iters, n_steps, 3))
  reloc_matrix <- matrix(0, nrow = n_iters, ncol = n_steps)

  for (i in seq_along(j)) {
    iter_df <- j[[i]]
    counts <- iter_df$per_governance_community_count
    count_matrix[i, , 1] <- counts$direct
    count_matrix[i, , 2] <- counts$coalition
    count_matrix[i, , 3] <- counts$algorithmic
    reloc_matrix[i, ] <- iter_df$n_relocations
  }

  # Compute mean + SE across iterations
  mean_counts <- apply(count_matrix, c(2, 3), mean, na.rm = TRUE)
  se_counts   <- apply(count_matrix, c(2, 3), function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x))))
  mean_reloc  <- colMeans(reloc_matrix, na.rm = TRUE)

  # Build long-format tibble
  steps <- seq_len(n_steps)
  bind_rows(
    tibble(step = steps, governance = "Direct",      mean_count = mean_counts[, 1], se_count = se_counts[, 1], mean_relocations = mean_reloc),
    tibble(step = steps, governance = "Coalition",   mean_count = mean_counts[, 2], se_count = se_counts[, 2], mean_relocations = mean_reloc),
    tibble(step = steps, governance = "Algorithmic", mean_count = mean_counts[, 3], se_count = se_counts[, 3], mean_relocations = mean_reloc)
  )
}

# ===========================================================================
# Load all data
# ===========================================================================

message("\n--- Loading governance trajectory data ---")

all_data <- list()
for (cfg in configs) {
  d <- load_gov_trajectory(cfg$dir)
  if (!is.null(d)) {
    d$config <- cfg$label_str
    all_data[[length(all_data) + 1]] <- d
  }
}

traj_data <- bind_rows(all_data)

if (nrow(traj_data) == 0) {
  stop("No trajectory data found. Exiting.")
}

config_order <- sapply(configs, function(x) x$label_str)
traj_data$config <- factor(traj_data$config, levels = config_order)
traj_data$governance <- factor(traj_data$governance, levels = gov_levels)

# ===========================================================================
# Figure 1: Stacked Area Chart
# ===========================================================================

message("\n--- Figure 1: Stacked Area Chart ---")

# Area chart panels
area_panels <- list()
reloc_panels <- list()

for (cfg in configs) {
  cfg_data <- traj_data %>% filter(config == cfg$label_str)

  p_area <- ggplot(cfg_data, aes(x = step, y = mean_count, fill = governance)) +
    geom_area(alpha = 0.85, colour = "white", linewidth = 0.3) +
    geom_hline(yintercept = 300, linetype = "dashed", colour = "grey40", linewidth = 0.4) +
    geom_vline(xintercept = c(20, 40, 60, 80), linetype = "dotted",
               colour = "grey60", linewidth = 0.3) +
    scale_fill_manual(name = "Governance", values = gov_colours) +
    scale_x_continuous(breaks = seq(0, 100, 20), limits = c(1, 100), expand = c(0, 0)) +
    coord_cartesian(ylim = c(0, 900)) +
    scale_y_continuous(breaks = seq(0, 900, 150), expand = c(0, 0)) +
    labs(x = NULL, y = "Community count",
         title = cfg$label) +
    theme(legend.position = "none",
          plot.title = element_text(size = 11, hjust = 0.5))

  # Relocation subplot
  reloc_data <- cfg_data %>%
    filter(governance == "Direct") %>%  # relocations same for all gov rows
    select(step, mean_relocations)

  p_reloc <- ggplot(reloc_data, aes(x = step, y = mean_relocations)) +
    geom_line(colour = "grey30", linewidth = 0.5) +
    geom_vline(xintercept = c(20, 40, 60, 80), linetype = "dotted",
               colour = "grey60", linewidth = 0.3) +
    scale_x_continuous(breaks = seq(0, 100, 20), limits = c(1, 100), expand = c(0, 0)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    labs(x = "Simulation step", y = "Relocations") +
    theme(axis.title.x = element_text(size = 10),
          axis.text = element_text(size = 8),
          plot.margin = margin(2, 8, 4, 8))

  area_panels[[length(area_panels) + 1]] <- p_area
  reloc_panels[[length(reloc_panels) + 1]] <- p_reloc
}

# Remove x-axis labels from top/middle area panels
area_panels[[1]] <- area_panels[[1]] + theme(axis.text.x = element_blank())
area_panels[[2]] <- area_panels[[2]] + theme(axis.text.x = element_blank())
# Only bottom relocation panel gets x-axis label
reloc_panels[[1]] <- reloc_panels[[1]] + labs(x = NULL) + theme(axis.text.x = element_blank())
reloc_panels[[2]] <- reloc_panels[[2]] + labs(x = NULL) + theme(axis.text.x = element_blank())

# Show legend only on the bottom area panel
area_panels[[3]] <- area_panels[[3]] + theme(legend.position = "bottom")

# Assemble: stack 6 panels (area, reloc) x 3 configs
p_fig1 <- wrap_plots(
  area_panels[[1]], reloc_panels[[1]],
  area_panels[[2]], reloc_panels[[2]],
  area_panels[[3]], reloc_panels[[3]],
  ncol = 1,
  heights = c(4, 1, 4, 1, 4, 1)
) + plot_annotation(
  title = "Governance community counts over simulation steps",
  subtitle = expression(rho[e] == 0.15 ~ " -- dashed line: equal split (300)"),
  theme = theme(
    plot.title    = element_text(size = 13, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 11)
  )
)

p_fig1_final <- p_fig1

save_fig(p_fig1_final, "fig_governance_flows_area", width = 8, height = 12)


# ===========================================================================
# Figure 2: Alluvial Diagram via ggforce::geom_parallel_sets
# ===========================================================================

message("\n--- Figure 2: Alluvial Diagram ---")

time_slices <- c(1, 20, 40, 60, 80, 100)
slice_labels <- paste0("s", time_slices)

# ---------------------------------------------------------------------------
# Flow decomposition algorithm
# ---------------------------------------------------------------------------

decompose_alluvial_flows <- function(traj_data_config) {
  # Extract mean counts at 6 time slices -> 6x3 matrix
  # Rows = time slices, Cols = Direct, Coalition, Algorithmic
  mat <- matrix(0, nrow = length(time_slices), ncol = 3,
                dimnames = list(slice_labels, gov_levels))

  for (g_idx in seq_along(gov_levels)) {
    g <- gov_levels[g_idx]
    g_data <- traj_data_config %>% filter(governance == g)
    for (t_idx in seq_along(time_slices)) {
      row <- g_data %>% filter(step == time_slices[t_idx])
      mat[t_idx, g_idx] <- round(row$mean_count)
    }
  }

  total <- sum(mat[1, ])  # should be ~900

  # Step 1: Base stays - minimum across all slices for each governance
  base_stays <- apply(mat, 2, min)
  rows <- list()
  row_id <- 0

  for (g_idx in seq_along(gov_levels)) {
    if (base_stays[g_idx] > 0) {
      row_id <- row_id + 1
      r <- setNames(rep(gov_levels[g_idx], length(time_slices)), slice_labels)
      rows[[row_id]] <- c(as.list(r), list(id = row_id, value = base_stays[g_idx]))
    }
  }

  # Step 2: Remaining after base
  remaining <- mat
  for (g_idx in seq_along(gov_levels)) {
    remaining[, g_idx] <- mat[, g_idx] - base_stays[g_idx]
  }

  # Step 3: Forward pass through transitions
  for (t in seq_len(length(time_slices) - 1)) {
    deltas <- remaining[t + 1, ] - remaining[t, ]
    losers  <- which(deltas < 0)
    gainers <- which(deltas > 0)

    if (length(losers) == 0 || length(gainers) == 0) next

    total_gain <- sum(deltas[gainers])

    for (l in losers) {
      loss <- abs(deltas[l])
      for (g in gainers) {
        transfer <- round(loss * deltas[g] / total_gain)
        if (transfer <= 0) next

        row_id <- row_id + 1
        r <- setNames(rep(NA_character_, length(time_slices)), slice_labels)
        # Source governance for slices 1..t
        for (s in 1:t) r[s] <- gov_levels[l]
        # Destination governance for slices t+1..end
        for (s in (t + 1):length(time_slices)) r[s] <- gov_levels[g]
        rows[[row_id]] <- c(as.list(r), list(id = row_id, value = transfer))
      }
    }

    # Update remaining: reduce losers, reduce gainers by what they received
    for (l in losers) {
      remaining[t + 1, l] <- 0
    }
    for (g in gainers) {
      # Gainers keep what they had + transfers (which are already in remaining)
      # Just zero out what was transferred from remaining tracking
    }
    # Recompute remaining from actual minus assigned
    # Actually, simpler: just recalculate remaining based on assigned so far
  }

  # Also add staying rows for remaining surplus at each slice
  # (communities that stay in their governance but above the base)
  for (g_idx in seq_along(gov_levels)) {
    min_remaining <- min(remaining[, g_idx])
    if (min_remaining > 0) {
      row_id <- row_id + 1
      r <- setNames(rep(gov_levels[g_idx], length(time_slices)), slice_labels)
      rows[[row_id]] <- c(as.list(r), list(id = row_id, value = min_remaining))
    }
  }

  if (length(rows) == 0) return(NULL)

  # Convert to data frame
  flow_df <- bind_rows(lapply(rows, as_tibble))
  flow_df$value <- as.numeric(flow_df$value)
  flow_df <- flow_df %>% filter(value > 0)

  # Adjust to ensure column sums match original totals
  # (Simple normalization: scale each slice's values to match mat totals)
  for (t_idx in seq_along(time_slices)) {
    sl <- slice_labels[t_idx]
    for (g in gov_levels) {
      current_sum <- sum(flow_df$value[flow_df[[sl]] == g], na.rm = TRUE)
      target <- mat[t_idx, g]
      if (current_sum > 0 && target > 0) {
        mask <- flow_df[[sl]] == g
        flow_df$value[mask] <- flow_df$value[mask] * target / current_sum
      }
    }
  }

  flow_df
}

# ---------------------------------------------------------------------------
# Build alluvial panels
# ---------------------------------------------------------------------------

alluvial_panels <- list()

for (cfg_idx in seq_along(configs)) {
  cfg <- configs[[cfg_idx]]
  cfg_data <- traj_data %>% filter(config == cfg$label_str)

  flow_df <- decompose_alluvial_flows(cfg_data)

  if (is.null(flow_df) || nrow(flow_df) == 0) {
    message("  Skipping alluvial for ", cfg$dir, ": no flow data")
    next
  }

  # Convert to long format for geom_parallel_sets
  # Need: id, dimension (time slice), value (governance), weight
  long_df <- flow_df %>%
    mutate(row_id = row_number()) %>%
    pivot_longer(cols = all_of(slice_labels),
                 names_to = "time_slice",
                 values_to = "governance") %>%
    mutate(
      time_slice = factor(time_slice, levels = slice_labels),
      governance = factor(governance, levels = gov_levels)
    )

  # Create the parallel sets data using gather_set_data pattern
  # ggforce needs: id, x (dimension index), y (category), weight
  long_df <- long_df %>%
    mutate(x = as.integer(time_slice))

  # Use gather_set_data for proper formatting
  set_data <- flow_df %>%
    mutate(row_id = row_number()) %>%
    gather_set_data(x = slice_labels)

  set_data$x <- factor(set_data$x, levels = slice_labels)
  set_data$y <- factor(set_data$y, levels = gov_levels)

  # Determine source governance (first slice) for ribbon colouring
  source_gov <- flow_df %>%
    mutate(row_id = row_number()) %>%
    select(row_id, source_gov = s1)
  set_data <- set_data %>%
    left_join(source_gov, by = c("id" = "row_id"))

  p <- ggplot(set_data, aes(x = x, id = id, split = y, value = value)) +
    geom_parallel_sets(aes(fill = source_gov), alpha = 0.5, axis.width = 0.15) +
    geom_parallel_sets_axes(axis.width = 0.15, fill = "grey80", colour = "grey50") +
    geom_parallel_sets_labels(colour = "black", size = 3, angle = 0) +
    scale_fill_manual(name = "Source", values = gov_colours) +
    scale_x_discrete(labels = paste0("Step ", time_slices)) +
    labs(x = NULL, y = "Communities", title = cfg$label) +
    theme(
      legend.position = "none",
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid = element_blank(),
      plot.title = element_text(size = 11, hjust = 0.5)
    )

  alluvial_panels[[length(alluvial_panels) + 1]] <- p
}

if (length(alluvial_panels) >= 3) {
  p_fig2 <- alluvial_panels[[1]] / alluvial_panels[[2]] / alluvial_panels[[3]] +
    plot_annotation(
      title = "Governance flow alluvial diagram",
      subtitle = expression(rho[e] == 0.15 ~ " -- ribbon colour: source governance"),
      theme = theme(
        plot.title    = element_text(size = 13, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 11)
      )
    ) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom") &
    scale_fill_manual(name = "Source governance", values = gov_colours)

  save_fig(p_fig2, "fig_governance_alluvial", width = 10, height = 12)
} else {
  message("  SKIPPING Figure 2: insufficient alluvial panels (", length(alluvial_panels), ")")
}

message("\n=== Governance flow figures generated ===")
