import pandas as pd
df = pd.read_csv("logs/history.csv")
# Find step where UAV was closest to N10.
# N10 coordinates are node_10_x and node_10_y.
# Let's print the columns first or print the rows where distance to N10 is small.
import numpy as np
n10_x = df["node_10_x"].iloc[0]
n10_y = df["node_10_y"].iloc[0]
df["dist_to_n10"] = np.sqrt((df["uav_x"] - n10_x)**2 + (df["uav_y"] - n10_y)**2)
closest_steps = df.sort_values(by="dist_to_n10").head(10)
print(closest_steps[["step", "uav_x", "uav_y", "dist_to_n10", "node_10_status", "node_10_aoi", "jammer_channel", "uav_channel"]])
