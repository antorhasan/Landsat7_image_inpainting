
import random
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

mean = np.load("/media/antor/Files/main_projects/gitlab/Landsat7_image_inpainting/train_mean.npy")

def createDataRecord(out_filename, addrs_y):
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs_y)):
        print(i)
        img_y = cv2.imread(addrs_y[i], cv2.IMREAD_GRAYSCALE)
        img_y = np.asarray(img_y)
        last_y = np.reshape(img_y, (256,256,1))
        last_y = last_y/255
        last_y = last_y-mean

        feature = {
            'image_y': _bytes_feature(last_y.tostring())
                    }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


trainY = "/media/antor/Files/ML/Papers/slc_inpainting/Data/last_gt/*.jpg"
trainY_list = glob.glob(trainY)


random.seed(3)
random.shuffle(trainY_list)

train_Y = trainY_list[0:19488]
val_Y = trainY_list[19488:21376]

createDataRecord("/media/antor/Files/ML/tfrecord/slc_inpainting/train_unet.tfrecords", train_Y)
createDataRecord("/media/antor/Files/ML/tfrecord/slc_inpainting/val_unet.tfrecords", val_Y)
