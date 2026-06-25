# Load necessary libraries
# Check if packages are installed, if not, install them
if (!require("tidyverse")) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require("jsonlite")) install.packages("jsonlite", repos = "http://cran.us.r-project.org")
if (!require("gridExtra")) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(jsonlite)
library(gridExtra)

# --- Configuration ---
# Set the base directory for experiments
# Check if "artifacts" exists in current dir (running from root), else try up (running from subdir)
if (dir.exists("artifacts")) {
  LOGS_DIR <- "artifacts"
} else {
  LOGS_DIR <- "../artifacts"
}

# Function to find the latest experiment directory
find_latest_experiment <- function(base_dir) {
  # List all directories in artifacts
  dirs <- list.dirs(base_dir, recursive = FALSE)

  if (length(dirs) == 0) {
    stop("No experiment directories found in ", base_dir)
  }

  # Sort by name (works for ISO timestamps)
  latest_dir <- tail(sort(dirs), 1)
  return(latest_dir)
}

# --- Main Visualization Logic ---

vis_simulation <- function(exp_dir = NULL) {
  # 1. Determine Experiment Directory
  if (is.null(exp_dir)) {
    exp_dir <- find_latest_experiment(LOGS_DIR)
    message("No experiment directory specified. Using latest: ", exp_dir)
  }

  # 2. Locate Data (history.csv)
  # Try direct path
  csv_path <- file.path(exp_dir, "history.csv")
  config_path <- file.path(exp_dir, "config.json")

  # If not found, check for deep RL structure: {algo}/evaluation/history.csv
  if (!file.exists(csv_path)) {
    message("history.csv not found in root. Checking subdirectories...")

    # Explicitly check for known algorithms
    algos <- c("ppo", "baseline", "dqn", "ppo_lstm")
    found_algo <- NULL

    for (algo in algos) {
      candidate <- file.path(exp_dir, algo, "evaluation", "history.csv")
      if (file.exists(candidate)) {
        csv_path <- candidate
        found_algo <- algo
        break
      }
    }

    # If still not found, try recursive search as last resort
    if (is.null(found_algo)) {
      candidates <- list.files(exp_dir, pattern = "history.csv", recursive = TRUE, full.names = TRUE)
      if (length(candidates) == 0) {
        # Debug info
        message("Contents of ", exp_dir, ":")
        print(list.files(exp_dir))
        stop("No history.csv found in ", exp_dir)
      }
      csv_path <- candidates[1]
    }

    message("Found data at: ", csv_path)

    # Update config path relative to csv
    csv_dir <- dirname(csv_path)
    # check standard locations from csv dir up
    promising_configs <- c(
      file.path(csv_dir, "config.json"),
      file.path(dirname(csv_dir), "config.json"), # ../config.json
      file.path(exp_dir, "config.json") # root/config.json
    )

    for (cp in promising_configs) {
      if (file.exists(cp)) {
        config_path <- cp
        break
      }
    }
  }

  # Set exp_dir to where we output images (csv_dir)
  output_dir <- dirname(csv_path)

  # 3. Load Data
  df <- read_csv(csv_path, show_col_types = FALSE)

  # Load Config
  config <- list()
  if (file.exists(config_path)) {
    config <- fromJSON(config_path)
    message("Config loaded from: ", config_path)
  } else {
    warning("config.json not found. Using defaults.")
  }

  message("Processing data for output in: ", output_dir)
  message("Data Loaded: ", nrow(df), " steps.")

  # 4. Preprocess Data

  # 4.1 Calculate SINR (dB)
  # Find 'node_X_sinr' columns
  sinr_cols <- grep("node_.*_sinr", names(df), value = TRUE)

  if (length(sinr_cols) > 0) {
    # Calculate row-wise mean of linear SINR
    df$avg_linear_sinr <- rowMeans(df[, sinr_cols], na.rm = TRUE)
    # Avoid log(0)
    df$avg_linear_sinr <- pmax(df$avg_linear_sinr, 1e-12)
    df$sinr_db <- 10 * log10(df$avg_linear_sinr)
  } else {
    df$sinr_db <- 10 * log10(pmax(df$sinr %||% 1e-12, 1e-12)) # Fallback if sinr col exists
    if (!("sinr_db" %in% names(df))) df$sinr_db <- 0
  }

  # 4.2 Calculate Avg AoI
  aoi_cols <- grep("node_.*_aoi", names(df), value = TRUE)
  if (length(aoi_cols) > 0) {
    df$aoi_avg <- rowMeans(df[, aoi_cols], na.rm = TRUE)
  } else {
    df$aoi_avg <- df$aoi %||% 0 # Fallback
  }

  # 4.3 Node Status (Connected, Jammed, Out of Range)
  # Status Mapping: 0=Connected, 1=Out of Range, 2=Jammed
  # We want a 'step_status' for the UAV (simulating the Python logic)
  status_cols <- grep("node_.*_status", names(df), value = TRUE)

  if (length(status_cols) > 0) {
    # Python Logic:
    # 1. Default to 1 (Out of Range)
    # 2. If ANY node is 2 (Jammed) -> 2
    # 3. If ANY node is 0 (Connected) -> 0 (Overwrites 2)

    df$step_status <- 1 # Default

    # Check if any is 2
    is_jammed <- apply(df[, status_cols] == 2, 1, any)
    df$step_status[is_jammed] <- 2

    # Check if any is 0
    is_connected <- apply(df[, status_cols] == 0, 1, any)
    df$step_status[is_connected] <- 0
  } else {
    # Fallback to jammed_count
    df$step_status <- 0
    if ("jammed_count" %in% names(df)) {
      df$step_status[df$jammed_count > 0] <- 2
    }
  }

  # Convert step_status to factor for plotting
  df$step_status_label <- factor(df$step_status,
    levels = c(0, 1, 2),
    labels = c("Connected", "Out of Range", "Jammed")
  )

  # 5. Generate Plots

  # Plot 1: Trajectory (Only if spatial data exists)
  if ("uav_x" %in% names(df)) {
    plot_trajectory(df, config, output_dir)
  }

  # Plot 2: Metrics (SINR, Jamming Power, AoI, Energy)
  plot_metrics(df, output_dir)

  # Plot 3: Advanced Metrics (Bar Charts)
  plot_advanced_metrics(df, output_dir)

  # Plot 4: Channel Usage
  plot_channel_usage(df, output_dir)

  message("All plots generated in ", output_dir)
}

# --- Plotting Functions ---

plot_trajectory <- function(df, config, exp_dir) {
  p <- ggplot() +
    theme_minimal() +
    labs(
      title = "UAV Trajectory & Network Status",
      x = "Position X (m)", y = "Position Y (m)"
    )

  # 1. IoT Nodes (Static - from Step 0)
  node_x_cols <- sort(grep("node_.*_x", names(df), value = TRUE))
  node_y_cols <- sort(grep("node_.*_y", names(df), value = TRUE))

  if (length(node_x_cols) > 0) {
    # Extract first row positions
    nodes_df <- data.frame(
      x = as.numeric(df[1, node_x_cols]),
      y = as.numeric(df[1, node_y_cols]),
      id = paste0("N", seq_along(node_x_cols) - 1)
    )

    p <- p + geom_point(
      data = nodes_df, aes(x = x, y = y),
      color = "forestgreen", shape = 15, size = 4
    ) +
      geom_text(
        data = nodes_df, aes(x = x + 10, y = y + 10, label = id),
        size = 3, fontface = "bold"
      )
  }

  # 2. UAV Path
  p <- p + geom_path(
    data = df, aes(x = uav_x, y = uav_y),
    color = "gray", linewidth = 1, alpha = 0.6
  )

  # 3. UAV States (Points on Path)
  # Color by status
  status_colors <- c("Connected" = "blue", "Out of Range" = "gray", "Jammed" = "red")

  p <- p + geom_point(
    data = df, aes(x = uav_x, y = uav_y, color = step_status_label, shape = step_status_label),
    size = 2, alpha = 0.8
  ) +
    scale_color_manual(values = status_colors, drop = FALSE) +
    scale_shape_manual(values = c(16, 1, 4), drop = FALSE)

  # 4. Start/End
  start_pt <- df[1, ]
  end_pt <- df[nrow(df), ]

  p <- p + geom_point(data = start_pt, aes(x = uav_x, y = uav_y), color = "blue", shape = 17, size = 5) + # Triangle
    geom_point(data = end_pt, aes(x = uav_x, y = uav_y), color = "blue", shape = 15, size = 4) # Square

  # 5. Attacker
  att_x <- as.numeric(config$ATTACKER_POS_X %||% 500)
  att_y <- as.numeric(config$ATTACKER_POS_Y %||% 500)

  p <- p + annotate("point", x = att_x, y = att_y, color = "darkred", shape = 8, size = 6) +
    annotate("text", x = att_x, y = att_y + 20, label = "Attacker", color = "darkred")

  # Limits
  area_size <- as.numeric(config$AREA_SIZE %||% 1000)
  p <- p + xlim(0, area_size) + ylim(0, area_size) + coord_fixed()

  ggsave(file.path(exp_dir, "trajectory_r.png"), plot = p, width = 8, height = 8, dpi = 300)
  ggsave(file.path(exp_dir, "trajectory_r.pdf"), plot = p, width = 8, height = 8)
}

plot_metrics <- function(df, exp_dir) {
  # Plot A: SINR
  p1 <- ggplot(df, aes(x = step)) +
    geom_line(aes(y = sinr_db), color = "#1f77b4", linewidth = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
    theme_bw() +
    labs(y = "SINR (dB)", x = NULL, title = "Communication Quality (SINR)")

  # Plot B: Jamming
  if ("jammer_power" %in% names(df)) {
    p2 <- ggplot(df, aes(x = step)) +
      geom_line(aes(y = jammer_power), color = "#d62728", linewidth = 1, linetype = "dotted") +
      theme_bw() +
      labs(y = "Power (W)", x = NULL, title = "Jamming Power")
  } else {
    p2 <- ggplot() +
      theme_void()
  }

  # Plot C: AoI
  p3 <- ggplot(df, aes(x = step, y = aoi_avg)) +
    geom_line(color = "forestgreen", linewidth = 1) +
    theme_bw() +
    labs(y = "Avg AoI (s)", x = NULL, title = "Age of Information")

  # Plot D: Energy
  if ("uav_energy" %in% names(df)) {
    p4 <- ggplot(df, aes(x = step, y = uav_energy)) +
      geom_line(color = "#ff7f0e", linewidth = 1) +
      theme_bw() +
      labs(y = "Energy (J)", x = "Time Step", title = "UAV Energy Consumption")
  } else {
    p4 <- ggplot() +
      theme_void()
  }

  # Combine
  p_combined <- grid.arrange(p1, p2, p3, p4, nrow = 4)

  ggsave(file.path(exp_dir, "metrics_analysis_r.png"), plot = p_combined, width = 8, height = 12, dpi = 300)
  ggsave(file.path(exp_dir, "metrics_analysis_r.pdf"), plot = p_combined, width = 8, height = 12)
}

plot_advanced_metrics <- function(df, exp_dir) {
  final_row <- tail(df, 1)

  # 1. Total Successful Duration
  total_time_cols <- grep("node_.*_total_time", names(df), value = TRUE)

  if (length(total_time_cols) > 0) {
    durations <- as.numeric(final_row[total_time_cols])
    node_ids <- paste0("N", seq_along(durations) - 1)

    vis_df <- data.frame(Node = node_ids, Duration = durations)

    p1 <- ggplot(vis_df, aes(x = Node, y = Duration)) +
      geom_bar(stat = "identity", fill = "skyblue", color = "black") +
      geom_text(aes(label = round(Duration, 1)), vjust = -0.5) +
      theme_minimal() +
      labs(title = "Total Successful Communication Duration", y = "Seconds")

    # 2. Max Continuous Duration
    max_cont_cols <- grep("node_.*_max_continuous_time", names(df), value = TRUE)
    if (length(max_cont_cols) > 0) {
      max_cont <- as.numeric(final_row[max_cont_cols])

      vis_df2 <- data.frame(Node = node_ids, Duration = max_cont)

      p2 <- ggplot(vis_df2, aes(x = Node, y = Duration)) +
        geom_bar(stat = "identity", fill = "salmon", color = "black") +
        geom_text(aes(label = round(Duration, 1)), vjust = -0.5) +
        theme_minimal() +
        labs(title = "Max Continuous Connection Streak", y = "Seconds")

      p_combined <- grid.arrange(p1, p2, nrow = 2)

      ggsave(file.path(exp_dir, "advanced_metrics_r.png"), plot = p_combined, width = 8, height = 10, dpi = 300)
      ggsave(file.path(exp_dir, "advanced_metrics_r.pdf"), plot = p_combined, width = 8, height = 10)
    }
  }
}

plot_channel_usage <- function(df, exp_dir) {
  if (!("uav_channel" %in% names(df)) || !("jammer_channel" %in% names(df))) {
    return()
  }

  # Identif collisions
  collisions <- df %>% filter(uav_channel == jammer_channel)

  p <- ggplot(df, aes(x = step)) +
    geom_step(aes(y = uav_channel, color = "UAV"), linewidth = 1) +
    geom_step(aes(y = jammer_channel, color = "Jammer"), linetype = "dashed", linewidth = 1) +
    scale_color_manual(values = c("UAV" = "blue", "Jammer" = "red")) +
    scale_y_continuous(breaks = 0:7, labels = paste("Ch", 0:7), limits = c(-0.5, 7.5)) +
    theme_minimal() +
    labs(title = "Channel Hopping Dynamics", x = "Time Step", y = "Channel", color = "Entity")

  if (nrow(collisions) > 0) {
    p <- p + geom_point(
      data = collisions, aes(x = step, y = uav_channel),
      shape = 21, size = 3, stroke = 1.5, color = "black", fill = "yellow"
    )
  }

  ggsave(file.path(exp_dir, "channel_usage_r.png"), plot = p, width = 10, height = 5, dpi = 300)
  ggsave(file.path(exp_dir, "channel_usage_r.pdf"), plot = p, width = 10, height = 5)
}

# Helper for default value
`%||%` <- function(a, b) if (!is.null(a)) a else b

# Execute
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  vis_simulation(args[1])
} else {
  vis_simulation()
}
