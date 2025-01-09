import cv2
import numpy as np

def label_areas(image,mask):
    
    labeled_img = np.zeros_like(image)
    labeled_img[image == 255] = 255
    road_colored = mask.copy()
    road_colored[labeled_img == 255] = [0, 255, 0]     # 將馬路區域著色為綠色
    return road_colored