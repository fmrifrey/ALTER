import numpy as np

data1 = np.load("abide.npy", allow_pickle=True).item()
data2 = np.load("abide_tmp.npy", allow_pickle=True).item()

print(data1["label"].shape)
print(data2["label"].shape)