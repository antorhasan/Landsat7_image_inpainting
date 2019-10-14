import tensorflow as tf
import numpy as np
import glob
import cv2
import random

trainY = "/media/antor/Files/ML/Papers/slc_inpainting/Data/last_gt/*.jpg"
trainY_list = glob.glob(trainY)

random.seed(3)
random.shuffle(trainY_list)
trainY_list = trainY_list[0:19488]
#trainY_list = trainY_list[0:10]

for i in range(len(trainY_list)):
    img_y = cv2.imread(trainY_list[i], cv2.IMREAD_GRAYSCALE)
    img_y = np.asarray(img_y)
    img_y = np.reshape(img_y,(256,256,1))
    img_y = img_y/255
    #img_y = np.expand_dims(img_y, axis=0)
    #print(img_y)
    print(i)

    if i==0 :
        add_mean = img_y/len(trainY_list)
    else:
        add_mean += img_y/len(trainY_list)

#add_mean = np.expand_dims(add_mean, axis=0)
print(add_mean)
print(add_mean.shape)
np.save("/media/antor/Files/main_projects/gitlab/Landsat7_image_inpainting/train_mean.npy", add_mean)
