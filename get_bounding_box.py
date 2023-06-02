import cv2
import os
import numpy as np
path_dataset = "D:\HKIV\cs114\gray_dataset"
kernel = np.ones((5, 5), np.uint8)
for num in os.listdir(path_dataset):
    path_foder_num = path_dataset + "/" + num
    for im in os.listdir(path_foder_num):
        path_img_name = path_foder_num + "/" + im
        img_gray = cv2.imread(path_img_name)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel, iterations = 5)
        contours, _ = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_contour = [cv2.contourArea(cnt) for cnt in contours]
        area_sort = np.argsort(area_contour)[::-1]
        x, y, w, h = cv2.boundingRect(contours[area_sort[0]])
        pre_image = img_gray[y:(y+h), x:(x+w)]
        pre_image = cv2.resize(pre_image, (25, 25))
        pre_image = cv2.copyMakeBorder(pre_image,3,3,3,3,cv2.BORDER_CONSTANT, value = 0)
        cv2.imwrite(path_img_name, pre_image)
