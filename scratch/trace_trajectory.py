import pandas as pd
df = pd.read_csv("logs/history.csv")
status_cols = [c for c in df.columns if "node_" in c and "_status" in c]
print("Steps with status changes or active nodes:")
for idx, row in df.iterrows():
    statuses = {col.split("_")[1]: int(row[col]) for col in status_cols if row[col] != 1}
    if statuses:
        print(f"Step {int(row['step'])}: UAV at ({row['uav_x']:.1f}, {row['uav_y']:.1f}), Active nodes status: {statuses}, Jammer channel: {int(row['jammer_channel'])}, UAV channel: {int(row['uav_channel'])}")
