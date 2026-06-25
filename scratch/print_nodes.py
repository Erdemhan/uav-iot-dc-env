import pandas as pd
df = pd.read_csv("logs/history.csv")
node_cols_x = sorted([c for c in df.columns if "node_" in c and "_x" in c])
node_cols_y = sorted([c for c in df.columns if "node_" in c and "_y" in c])
print("Node Coordinates:")
for cx, cy in zip(node_cols_x, node_cols_y):
    node_id = cx.split("_")[1]
    print(f"Node {node_id}: ({df[cx].iloc[0]:.2f}, {df[cy].iloc[0]:.2f})")
