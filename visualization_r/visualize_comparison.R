# Check if packages are installed, if not, install them
if (!require("tidyverse")) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require("jsonlite")) install.packages("jsonlite", repos = "http://cran.us.r-project.org")
if (!require("gridExtra")) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(jsonlite)
library(gridExtra)

# --- Configuration ---
# Default directory (can be overridden)
if (dir.exists("artifacts")) {
  DEFAULT_RUN_DIR <- "artifacts"
} else {
  DEFAULT_RUN_DIR <- "../artifacts"
}

# --- Data Loading & Preprocessing ---

load_algo_data <- function(run_dir, algo_name, label) {
  # Path structure: run_dir/{algo_name}/evaluation/history.csv
  csv_path <- file.path(run_dir, algo_name, "evaluation", "history.csv")

  if (!file.exists(csv_path)) {
    warning(paste("Data not found for:", algo_name, "at", csv_path))
    return(NULL)
  }

  df <- read_csv(csv_path, show_col_types = FALSE)

  # Preprocess
  # 1. SINR dB
  sinr_cols <- grep("node_.*_sinr", names(df), value = TRUE)
  if (length(sinr_cols) > 0) {
    df$avg_linear_sinr <- rowMeans(df[, sinr_cols], na.rm = TRUE)
    df$avg_linear_sinr <- pmax(df$avg_linear_sinr, 1e-12)
    df$avg_sinr_db <- 10 * log10(df$avg_linear_sinr)
  } else {
    df$avg_sinr_db <- 0
  }

  # 2. Jamming Success Rate
  status_cols <- grep("node_.*_status", names(df), value = TRUE)
  if (length(status_cols) > 0) {
    # Count how many are Jammed (2)
    jammed_counts <- rowSums(df[, status_cols] == 2, na.rm = TRUE)
    df$jammed_count <- jammed_counts
    df$jam_success_rate <- jammed_counts / length(status_cols)
  } else {
    # Fallback to existing jammed_count if available
    if ("jammed_count" %in% names(df)) {
      df$jam_success_rate <- df$jammed_count / 10.0 # Assuming 10 nodes
    } else {
      df$jam_success_rate <- 0
    }
  }

  # 3. Channel Match
  if ("uav_channel" %in% names(df) & "jammer_channel" %in% names(df)) {
    df$channel_match <- as.integer(df$uav_channel == df$jammer_channel)
    # Rolling average for smooth plot (window=5)
    df$channel_match_smooth <- stats::filter(df$channel_match, rep(1 / 5, 5), sides = 1)
  } else {
    df$channel_match <- 0
    df$channel_match_smooth <- 0
  }

  # Add Metadata
  df$Algorithm <- label
  df$AlgoKey <- algo_name

  return(df)
}

# --- Main Comparison Logic ---

compare_experiments <- function(run_dir = NULL) {
  if (is.null(run_dir)) {
    # Use the configured artifacts root to find latest experiment
    dirs <- list.dirs(DEFAULT_RUN_DIR, recursive = FALSE)
    if (length(dirs) == 0) {
      stop("No experiment directories found in ", DEFAULT_RUN_DIR)
    }
    run_dir <- tail(sort(dirs), 1)
    message("No run directory specified. Using latest: ", run_dir)
  }

  # Define Algorithms to Compare
  algos <- list(
    list(name = "baseline", label = "Baseline (QJC)", color = "gray"),
    list(name = "ppo", label = "Deep RL (PPO)", color = "#1f77b4"), # Tab:blue
    list(name = "dqn", label = "Deep RL (DQN)", color = "#ff7f0e"), # Tab:orange
    list(name = "ppo_lstm", label = "Deep RL (PPO-LSTM)", color = "#9467bd") # Tab:purple
  )

  all_data <- list()

  for (algo in algos) {
    df <- load_algo_data(run_dir, algo$name, algo$label)
    if (!is.null(df)) {
      all_data[[algo$label]] <- df
    }
  }

  if (length(all_data) == 0) {
    stop("No data found for any algorithm in ", run_dir)
  }

  # Combine Data for Plotting
  combined_df <- bind_rows(all_data)

  # Fix factor order for legend
  combined_df$Algorithm <- factor(combined_df$Algorithm, levels = sapply(algos, function(x) x$label))

  # Create Output Directory
  output_dir <- file.path(run_dir, "comparison_r")
  dir.create(output_dir, showWarnings = FALSE)

  # --- Visualization ---

  # --- Enhanced Visualizations ---

  # Plot 1: Jamming Success Rate (with Smoothing)
  p1 <- ggplot(combined_df, aes(x = step, y = jam_success_rate * 100, color = Algorithm, fill = Algorithm)) +
    geom_line(alpha = 0.3, linewidth = 0.5) + # Faint raw data
    geom_smooth(method = "loess", se = FALSE, linewidth = 1.2) + # Bold smooth trend
    scale_color_manual(values = setNames(sapply(algos, function(x) x$color), sapply(algos, function(x) x$label))) +
    scale_fill_manual(values = setNames(sapply(algos, function(x) x$color), sapply(algos, function(x) x$label))) +
    theme_minimal() +
    labs(title = "Jamming Success Rate", subtitle = "(Higher is Better)", x = "Time Step", y = "Success Rate (%)") +
    theme(legend.position = "top")

  # Plot 2: Network SINR (with Smoothing)
  p2 <- ggplot(combined_df, aes(x = step, y = avg_sinr_db, color = Algorithm, fill = Algorithm)) +
    geom_line(alpha = 0.3, linewidth = 0.5) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 1.2) +
    scale_color_manual(values = setNames(sapply(algos, function(x) x$color), sapply(algos, function(x) x$label))) +
    theme_minimal() +
    labs(title = "Network SINR", subtitle = "(Lower is Better - Target Jammed)", x = "Time Step", y = "SINR (dB)") +
    theme(legend.position = "none")

  # Plot 3: Channel Tracking (with Smoothing)
  p3 <- ggplot(combined_df, aes(x = step, y = channel_match_smooth * 100, color = Algorithm, fill = Algorithm)) +
    geom_line(linewidth = 0.8, alpha = 0.6) + # It's already smoothed, keep as line but nicer
    scale_color_manual(values = setNames(sapply(algos, function(x) x$color), sapply(algos, function(x) x$label))) +
    theme_minimal() +
    labs(title = "Channel Tracking Accuracy", subtitle = "(Higher is Better)", x = "Time Step", y = "Match Rate (%)") +
    theme(legend.position = "none")

  # Plot 4: Training (Smoothed)
  # ... (Keep loading logic same, just update plot) ...
  train_data_list <- list()
  for (algo in algos) {
    # Baseline
    if (algo$name == "baseline") {
      train_path <- file.path(run_dir, "baseline", "training_curve.csv")
      if (file.exists(train_path)) {
        df_t <- read_csv(train_path, show_col_types = FALSE)
        BATCH_SIZE <- 1000
        df_t$total_steps <- df_t$episode * 100
        df_t$step_bin <- ((df_t$total_steps - 1) %/% BATCH_SIZE) * BATCH_SIZE + BATCH_SIZE
        df_resampled <- df_t %>%
          group_by(step_bin) %>%
          summarise(reward = mean(total_reward, na.rm = TRUE)) %>%
          mutate(Algorithm = algo$label)
        train_data_list[[algo$label]] <- df_resampled
      }
    } else {
      base_algo_dir <- file.path(run_dir, algo$name)
      if (dir.exists(base_algo_dir)) {
        progress_files <- list.files(base_algo_dir, pattern = "progress.csv", recursive = TRUE, full.names = TRUE)
        if (length(progress_files) > 0) {
          latest_file <- progress_files[which.max(file.info(progress_files)$mtime)]
          df_t <- read_csv(latest_file, show_col_types = FALSE)
          if ("timesteps_total" %in% names(df_t) && "env_runners/episode_reward_mean" %in% names(df_t)) {
            df_t <- df_t %>%
              select(step_bin = timesteps_total, reward = `env_runners/episode_reward_mean`) %>%
              mutate(Algorithm = algo$label)
            train_data_list[[algo$label]] <- df_t
          } else if ("timesteps_total" %in% names(df_t) && "episode_reward_mean" %in% names(df_t)) {
            df_t <- df_t %>%
              select(step_bin = timesteps_total, reward = episode_reward_mean) %>%
              mutate(Algorithm = algo$label)
            train_data_list[[algo$label]] <- df_t
          }
        }
      }
    }
  }

  if (length(train_data_list) > 0) {
    train_df <- bind_rows(train_data_list)
    train_df$Algorithm <- factor(train_df$Algorithm, levels = sapply(algos, function(x) x$label))

    p4 <- ggplot(train_df, aes(x = step_bin, y = reward, color = Algorithm)) +
      geom_point(size = 0.5, alpha = 0.2) +
      geom_smooth(se = FALSE, span = 0.3, linewidth = 1.2) +
      scale_color_manual(values = setNames(sapply(algos, function(x) x$color), sapply(algos, function(x) x$label))) +
      theme_minimal() +
      labs(title = "Training Progress", subtitle = "(Higher is Better)", x = "Total Steps", y = "Reward") +
      theme(legend.position = "none") +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray")
  } else {
    p4 <- ggplot() +
      theme_void() +
      labs(title = "No Training Data")
  }

  # --- Aggregated Stats & Bar Chart (Replacing Radar) ---
  stats_long <- combined_df %>%
    group_by(Algorithm) %>%
    summarise(
      `Success (%) (High is Better)` = mean(jam_success_rate, na.rm = TRUE) * 100,
      `Power (W) (Low is Better)` = mean(jammer_power, na.rm = TRUE),
      `Match (%) (High is Better)` = mean(channel_match, na.rm = TRUE) * 100,
      `SINR (dB) (Low is Better)` = mean(avg_sinr_db, na.rm = TRUE)
    ) %>%
    pivot_longer(cols = -Algorithm, names_to = "Metric", values_to = "Value")

  # Metrics Bar Chart
  p_metrics <- ggplot(stats_long, aes(x = Algorithm, y = Value, fill = Algorithm)) +
    geom_col(alpha = 0.8, width = 0.7) +
    facet_wrap(~Metric, scales = "free_y", nrow = 2) +
    scale_fill_manual(values = setNames(sapply(algos, function(x) x$color), sapply(algos, function(x) x$label))) +
    theme_minimal() +
    labs(title = "Performance Metrics Summary", x = NULL, y = NULL) +
    theme(
      legend.position = "none",
      axis.text.x = element_blank(), # Remove x-axis labels if legend is present elsewhere, or keep? Let's remove to save space.
      panel.grid.major.x = element_blank()
    )

  # Prepare Statistics Table for Display
  stats_display <- combined_df %>%
    group_by(Algorithm) %>%
    summarise(
      `Success (%)` = round(mean(jam_success_rate, na.rm = TRUE) * 100, 1),
      `Power (W)` = round(mean(jammer_power, na.rm = TRUE), 2),
      `Match (%)` = round(mean(channel_match, na.rm = TRUE) * 100, 1),
      `SINR (dB)` = round(mean(avg_sinr_db, na.rm = TRUE), 2)
    )

  tt <- ttheme_default(core = list(bg_params = list(fill = "white")), colhead = list(bg_params = list(fill = "#f0f0f0")))
  stats_grob <- tableGrob(stats_display, rows = NULL, theme = tt)

  # --- Layout ---

  # Legend Extraction
  p_legend_src <- ggplot(combined_df, aes(x = step, y = avg_sinr_db, color = Algorithm)) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = setNames(sapply(algos, function(x) x$color), sapply(algos, function(x) x$label))) +
    theme_minimal() +
    theme(legend.position = "bottom", legend.box = "horizontal", legend.background = element_rect(fill = "white", color = "gray90"), legend.key = element_rect(fill = "white")) +
    guides(color = guide_legend(nrow = 1))

  get_legend <- function(myggplot) {
    tmp <- ggplot_gtable(ggplot_build(myggplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    if (length(leg) > 0) {
      return(tmp$grobs[[leg]])
    }
    return(NULL)
  }
  legend <- get_legend(p_legend_src)

  # Arrange:
  # Row 1: Success | SINR
  # Row 2: Match | Training
  # Row 3: Bar Chart | Table
  # Row 4: Legend

  row1 <- arrangeGrob(p1, p2, nrow = 1)
  row2 <- arrangeGrob(p3, p4, nrow = 1)
  row3 <- arrangeGrob(p_metrics, stats_grob, nrow = 1)

  layout_list <- list(row1, row2, row3)
  heights_list <- c(10, 10, 10)

  if (!is.null(legend)) {
    layout_list[[4]] <- legend
    heights_list <- c(10, 10, 10, 1)
  }

  # Use arrangeGrob for proper ggsave object construction
  p_final <- arrangeGrob(grobs = layout_list, nrow = length(layout_list), heights = heights_list)

  # Save Plot - Use NEW filename to ensure visibility
  ggsave(file.path(output_dir, "comparison_dashboard.png"), plot = p_final, width = 14, height = 18, dpi = 300, bg = "white")
  ggsave(file.path(output_dir, "comparison_dashboard.pdf"), plot = p_final, width = 14, height = 18, bg = "white")

  write_csv(stats_display, file.path(output_dir, "comparison_dashboard_stats.csv"))
  print(stats_display)
  message("\n--- SUCCESS ---")
  message("New dashboard saved to: ", file.path(output_dir, "comparison_dashboard.png"))
}

# --- Execute if script ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  compare_experiments(args[1])
} else {
  # Default behavior: compare latest if run interactively, or wait for call
  compare_experiments()
}
