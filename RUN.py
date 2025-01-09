import cv2
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from PIL import Image
import soble
import hsv
import lbp
import histogram 
import one_norm_dist
import labeling
def main():
    radius = 1
    n_points = 8  
    patch_size = 10  
    # 讀取影像
    image = cv2.imread("C:/Users/Zephyrus/Documents/road/Image.jpg")
    #image_gray = cv2.imread('Image.jpg',0)
    
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #sobel
    Sobel_img = soble.sobel(image_gray)
    cv2.imwrite('sobel.jpg',Sobel_img)

    #lbp
    lbp_img = lbp.lbp(Sobel_img)
    cv2.imwrite('lbp.jpg',lbp_img)
    
    #HSV
    hsv_img,hsv1 = hsv.hsv(image, Sobel_img, lbp_img)
    cv2.imwrite('hsv.jpg',hsv_img)
    hsv_lbp_img = lbp.lbp(hsv1)
    cv2.imwrite('hsv_lbp.jpg',hsv_lbp_img)
    #histogram
    mask = np.zeros_like(image_gray)  
    #histogram找前三大(設定閥值)
    his = histogram.calculate_histogram(hsv_lbp_img)
    hist = histogram.plot_histogram(his)
    top3 = histogram.find_top_three(his)
    th = int(sum(top3)/3)
    
    #BFS
    rows, cols = hsv_lbp_img.shape
    for i in range(0, rows, patch_size):
        for j in range(0, cols, patch_size):
            patch1 = hsv_lbp_img[i:i+patch_size, j:j+patch_size]
            if i+patch_size < rows and j+patch_size < cols:
                # 與右邊的區塊進行比較
                patch2 = hsv_lbp_img[i:i+patch_size, j+patch_size:j+2*patch_size]
                hist1 = histogram.calculate_histogram(patch1)
                hist2 = histogram.calculate_histogram(patch2)               
                if one_norm_dist.calculate_1_norm_distance(hist1, hist2) <= th:
                    mask[i:i+patch_size, j:j+patch_size] = 1  
    #labeling
    colored_img = labeling.label_areas(hsv_img,image)

    cv2.imwrite('final.jpg',colored_img)    
    
    # 顯示
    # 原始影像和著色後的影像
    cv2.imshow('Colored Image', colored_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()