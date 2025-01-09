import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt 
from skimage import exposure

def lbp(image):
    lbp = np.zeros_like(image, dtype=np.uint8)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            center = image[i, j]
            binary_str = ''.join(['1' if image[i + dx, j + dy] >= center else '0' 
                                  for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]])
            lbp[i, j] = int(binary_str, 2)
    return lbp