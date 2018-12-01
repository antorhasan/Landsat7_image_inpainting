import tensorflow as tf
import numpy as np
import glob
import cv2


trainY = "/media/antor/Files/ML/Papers/slc_inpainting/Data/last_gt/*.jpg"
trainY_list = glob.glob(trainY)

trainY_list = trainY_list[0:3]


for i in range(len(trainY_list)):
    img_y = cv2.imread(trainY_list[i], cv2.IMREAD_GRAYSCALE)
    img_y = np.asarray(img_y)
    img_y = np.reshape(img_y,(256,256,1))
    img_y = img_y/255
    img_y = np.expand_dims(img_y, axis=0)
    print(img_y)

    if i==0 :
        app_full = img_y
    else:

        app_full = np.append(app_full, img_y, axis=0)
    



print(app_full)
