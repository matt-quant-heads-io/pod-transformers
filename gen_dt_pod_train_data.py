from collections import deque
import os
import pandas as pd
import numpy as np


obs_size = 5
data_size = 50

pod_root_path = f'dt_train_data/zelda/pod_exp_traj_obs_5_ep_len_77_goal_size_50'

tile_map = {
    0: "empty",
    1: "solid",
    2: "player",
    3: "key",
    4: "door",
    5: "bat",
    6: "scorpion",
    7: "spider"
}

dfs = []
X = []
y = []

print(os.listdir(pod_root_path))

for file in os.listdir(pod_root_path):
    df = pd.read_csv(f"{pod_root_path}/{file}")
    dfs.append(df)

df = pd.concat(dfs)
print(f"df shape: {df.shape}")
df.head()