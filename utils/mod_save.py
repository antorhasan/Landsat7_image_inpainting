import numpy as np
import tensorflow as tf
import cv2
from tensorflow.python.framework import ops
from os import listdir
from os.path import isfile, join
from datetime import datetime


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")    #for tensorboard
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
root_logdir_m = "tf_models"
logdir_m = "{}/run-{}/".format(root_logdir_m, now)
root_logdir_v = "tf_vals"
logdir_v = "{}/run-{}/".format(root_logdir_v, now)


def _parse_function(example_proto):

    features = {
                "image_y": tf.FixedLenFeature((), tf.string )
                }

    parsed_features = tf.parse_single_example(example_proto, features)

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float64)

    image_y = tf.reshape(image_y, [256,256,1])

    image_y = tf.cast(image_y,dtype=tf.float32)

    return image_y



dataset = tf.data.TFRecordDataset("/media/antor/Files/ML/tfrecord/slc_inpainting/train_unet.tfrecords")
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(3000)
dataset = dataset.batch(16)
#dataset = dataset.repeat(num_epochs)
handle1 = tf.placeholder(tf.string)

iterator = tf.data.Iterator.from_string_handle(handle1, dataset.output_types)


#iterator = dataset.make_initializable_iterator(shared_name="iter")

pix_gt1 = iterator.get_next()
training_iterator = dataset.make_initializable_iterator()


pix_gt = tf.reshape(pix_gt1,[16,256,256,1])

signature_key = 'test_signature'
input_key = 'input_x'
#han_key = 'input_h'
outp1 = 'output1'
outp2 = 'output2'
outp3 = 'output3'

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
sess.run(training_iterator.initializer)
#sess.run(validation_iterator.initializer)
training_handle = sess.run(training_iterator.string_handle())
per_in = sess.run(pix_gt,feed_dict={handle1 : training_handle})
print(per_in)
#sess.run(iterator.initializer)


meta_graph_def = tf.saved_model.loader.load(sess, ['test_saved_model'], "/media/antor/Files/main_projects/gitlab/Landsat7_image_inpainting/utils/tf_models/run-20181203092053/")
signature = meta_graph_def.signature_def

x_tensor_name = signature[signature_key].inputs[input_key].name
# handle_name = signature[signature_key].inputs[han_key].name
out1 = signature[signature_key].outputs[outp1].name
out2 = signature[signature_key].outputs[outp2].name
out3 = signature[signature_key].outputs[outp3].name


x = sess.graph.get_tensor_by_name(x_tensor_name)
# h = sess.graph.get_tensor_by_name(handle_name)
o1 = sess.graph.get_tensor_by_name(out1)
o2 = sess.graph.get_tensor_by_name(out2)
o3 = sess.graph.get_tensor_by_name(out3)

per_in = sess.run(pix_gt,feed_dict={handle1 : training_handle})
# print(sess.run([o1,o2,o3], feed_dict={x:per_in, h : training_handle}))
print(sess.run([o1,o2,o3], feed_dict={x:per_in}))

sess.close()
