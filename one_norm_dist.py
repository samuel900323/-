import numpy as np

def calculate_1_norm_distance(hist1, hist2):
    # 計算一階曼哈頓距離
    distance = np.sum(np.abs(hist1 - hist2))
    return distance