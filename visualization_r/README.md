# Visualization with R

This folder contains R scripts to generate scientific-quality visualizations for the UAV-IoT Project. It uses `ggplot2` to create plots that are suitable for academic papers.

## Prerequisites

1.  **Install R**: Download and install R from [CRAN](https://cran.r-project.org/).
2.  **Install RStudio (Optional)**: Recommended for a better development experience.
3.  **Install Required Packages**:
    Open R or RStudio and run the following command to install the dependencies:

    ```r
    install.packages(c("tidyverse", "jsonlite", "gridExtra"))
    ```

## Usage

1.  **Navigate to the project root** (or this folder) in your terminal or R console.
2.  **Run the script**:
    The script automatically finds the *latest* experiment folder in `../artifacts` (relative to this folder) and generates plots there.

    **From Terminal:**
    ```bash
    Rscript visualization_r/visualize_results.R
    ```

    **From RStudio:**
    Open `visualize_results.R`, set the working directory to the script location (Session -> Set Working Directory -> To Source File Location), and source the file.

## Output

The script generates the following files in the experiment folder (e.g., `artifacts/EXP_YYYY.../`):

-   `trajectory_r.png` / `.pdf`: UAV path, node positions, and status (Connected/Jammed).
-   `metrics_analysis_r.png` / `.pdf`: Time-series of SINR, Jamming Power, AoI, and Energy.
-   `advanced_metrics_r.png` / `.pdf`: Bar charts for connection duration and streaks.
-   `channel_usage_r.png` / `.pdf`: Channel hopping timeline and collisions.

## Customization

To visualize a specific experiment, modify the last line of `visualize_results.R`:

```r
# Default (Latest)
# vis_simulation()

# Specific Folder
vis_simulation("../artifacts/2026-02-14_12-00-00_EXP")
```

## Comparison Visualization

To compare multiple algorithms (Baseline, PPO, DQN, PPO-LSTM) within a single run directory:

1.  **Usage**:
    ```bash
    Rscript visualization_r/visualize_comparison.R [optional_path_to_run_dir]
    ```
    If no path is provided, it attempts to use the latest run in `../artifacts`.

2.  **Output**:
    Creates a `comparison_r` folder inside the run directory containing:
    -   `comparison_plot_r.png`: Comparison curves for Accuracy, SINR, etc.
    -   `comparison_stats.csv`: Statistical summary table.

