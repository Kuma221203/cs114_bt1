import cv2
import os
import numpy as np
import shutil
X = []
y = []
kernel = np.ones((5, 5), np.uint8)
path = "D:\HKIV\cs114\Datasets"
# for name in os.listdir(path):
name = "3"
path_numbers = path + "/" + name
for img_name in os.listdir(path_numbers):
    path_img = path_numbers + "/" + img_name
    img_root = cv2.imread(path_img)
    img_root = cv2.resize(img_root, (280, 280))
    lab = cv2.cvtColor(img_root, cv2.COLOR_BGR2Lab)
    l_channel, a, b = cv2.split(lab)
    clade = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clade.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 125, 257, cv2.THRESH_BINARY_INV)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 3)

    # contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # area_contour = [cv2.contourArea(cnt) for cnt in contours]
    # area_sort = np.argsort(area_contour)[::-1]
    # x, y, w, h = cv2.boundingRect(contours[area_sort[0]])
    # img_bound = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    # egded = cv2.Canny(img, 30, 200)
    cv2.imshow(img_name, img)
    k = cv2.waitKey()
    if(k == 32):
        path_gray = path_img.replace("Datasets", "gray_dataset")
        path_predataset = path_img.replace("Datasets", "pre_dataset")
        cv2.imwrite(path_gray, img)
        os.rename(path_img, path_predataset)
        # os.replace(path_img, path_predataset)
        # shutil.move(path_img, path_predataset)
    cv2.destroyAllWindows()
    # img = cv2.erode(img_ss2, kernel, iterations = 1)
    # closing = cv2.morphologyEx(img_ss2, cv2.MORPH_CLOSE, kernel, iterations = 2)
    # closing = cv2.dilate(closing, kernel, iterations = 1)
    # img_reshape = cv2.resize(closing, (28, 28))
    # X_number = np.array(img_reshape).reshape((784))
    # X.append(X_number)
    # y.append(int(name))

# X = np.array(X)
# y = np.array(y)
# print(X.shape)
# print(y.shape)