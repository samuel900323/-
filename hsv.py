import cv2
import numpy as np
from skimage.feature import local_binary_pattern 
from collections import deque

def hsv(image, lbp_image, sobel_image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])  
    upper_black = np.array([180, 255, 80])  
    mask_black = cv2.inRange(hsv_image, lower_black, upper_black)
    lower_green = np.array([35, 50, 50]) 
    upper_green = np.array([85, 255, 255]) 
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    lower_dry_vegetation = np.array([15, 50, 50])  
    upper_dry_vegetation = np.array([35, 255, 255]) 
    mask_dry_vegetation = cv2.inRange(hsv_image, lower_dry_vegetation, upper_dry_vegetation)

    # 天空
    lower_blue = np.array([100, 50, 50])  # 藍色的低範圍
    upper_blue = np.array([140, 255, 255])  # 藍色的高範圍
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_exclude = cv2.bitwise_or(mask_green, mask_blue)
    mask_exclude = cv2.bitwise_or(mask_exclude, mask_dry_vegetation)
    mask_filtered = cv2.bitwise_and(mask_black, ~mask_exclude)
    combined = cv2.bitwise_or(lbp_image, sobel_image)  

    combined_filtered = cv2.bitwise_and(combined, combined, mask=mask_filtered)

    # 二值化
    _, binary = cv2.threshold(combined_filtered, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8) 
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) 
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    min_area = 2800  # 設置最小區域大小
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
           binary[labels == i] = 0  # 去除小區域

    # 使用形態學操作進一步填補小洞
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary, mask_filtered