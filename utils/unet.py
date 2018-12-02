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

def _parse_function(example_proto):

    features = {
                "image_y": tf.FixedLenFeature((), tf.string )
                }

    parsed_features = tf.parse_single_example(example_proto, features)

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float64)

    image_y = tf.reshape(image_y, [256,256,1])

    image_y = tf.cast(image_y,dtype=tf.float32)

    return image_y


def conv_block(pixel,kernel_size,filter_numbers,stride,nonlinearity,conv_t):
    with tf.name_scope("conv") as scope:
        kernel_h = kernel_size[0]
        kernel_w = kernel_size[1]

        if conv_t== "conv":
            kernel_d = pixel.get_shape().as_list()[3]
            kernel_o = filter_numbers
        if conv_t == "dconv":
            kernel_d = filter_numbers
            kernel_o = pixel.get_shape().as_list()[3]

        W = tf.get_variable('Weights', (kernel_h, kernel_w, kernel_d, kernel_o),
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        if conv_t == "conv":
            conv_out = tf.nn.conv2d(pixel, W, strides=stride, padding="VALID", name="conv")
        if conv_t == "dconv":
            out_shape_list = pixel.get_shape().as_list()
            out_shape_list[1] = ((pixel.get_shape().as_list()[1] + 1) * 2)-1
            out_shape_list[2] = ((pixel.get_shape().as_list()[2] + 1) * 2)-1
            out_shape_list[3] = filter_numbers
            out_shape = tf.constant(out_shape_list)
            conv_out = tf.nn.conv2d_transpose(pixel,W,out_shape,stride,padding="VALID",name="dconv")

        B = tf.get_variable('Biases',(1,1,1,conv_out.get_shape()[3]),
                            initializer=tf.constant_initializer(.01))

        normalized_out = tf.add(conv_out,B)

        if nonlinearity=="relu":
            up_pixel = tf.nn.relu(normalized_out, name="relu")
        elif nonlinearity=="leaky_relu":
            up_pixel = tf.nn.leaky_relu(normalized_out, name="leaky_relu")
        elif nonlinearity=="none":
            up_pixel = tf.nn.sigmoid(normalized_out, name="sigmoid")

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", B)
        tf.summary.histogram("activations", up_pixel)

        return up_pixel


def near_up_sampling(pixel, mask, output_size):
    with tf.name_scope("nearest_up") as scope:
        up_pixel = tf.image.resize_nearest_neighbor(pixel, size=output_size, name="nearest_pixel_up")
        up_mask = tf.image.resize_nearest_neighbor(pixel, size=output_size, name="nearest_mask_up")
        return up_pixel, up_mask

def concat(near_pixel, pconv_pixel, near_mask, pconv_mask):
    with tf.name_scope("concatenation") as scope:
        up_pixel = tf.concat([pconv_pixel, near_pixel], axis=3)
        up_mask = tf.concat([pconv_mask,near_mask], axis=3)
        return up_pixel, up_mask

def decoding_layer(pixel_in,mask_in,is_training, output_size_in, pconv_pixel1, pconv_mask1, filter_numbers1):
    with tf.name_scope("decoding") as scope:
        near_pixel1,near_mask1 = near_up_sampling(pixel_in,mask_in,output_size_in)
        concat_pixel,concat_mask = concat(near_pixel1, pconv_pixel1, near_mask1, pconv_mask1)
        pixel_out,mask_out = partial_conv(concat_pixel,concat_mask,is_training,[3,3],filter_numbers1,[1,1,1,1],
                                        True,"leaky_relu",trans=True)
        return pixel_out,mask_out

def forward_prop(is_training, pixel, mask):
    non_lin = "relu"

    with tf.variable_scope("PConv1") as scope:
        p_out2,m_out2 = partial_conv(pixel,mask,is_training,kernel_size=[3,3],filter_numbers=4,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv2") as scope:
        p_out3,m_out3 = partial_conv(p_out2,m_out2,is_training,kernel_size=[3,3],filter_numbers=8,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv3") as scope:
        p_out4,m_out4 = partial_conv(p_out3,m_out3,is_training,kernel_size=[3,3],filter_numbers=16,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv4") as scope:
        p_out5,m_out5 = partial_conv(p_out4,m_out4,is_training,kernel_size=[3,3],filter_numbers=16,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv5") as scope:
        p_out6,m_out6 = partial_conv(p_out5,m_out5,is_training,kernel_size=[3,3],filter_numbers=32,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv6") as scope:
        p_out7,m_out7 = partial_conv(p_out6,m_out6,is_training,kernel_size=[3,3],filter_numbers=32,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv7") as scope:
        p_out8,m_out8 = partial_conv(p_out7,m_out7,is_training,kernel_size=[3,3],filter_numbers=32,stride=[1,1,1,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("decoding8") as scope:
        p_out9,m_out9 = decoding_layer(p_out8,m_out8,is_training,(p_out7.get_shape().as_list()[1],p_out7.get_shape().as_list()[2]),
                                        p_out7,m_out7,filter_numbers1=32)

    with tf.variable_scope("decoding9") as scope:
        p_out10,m_out10 = decoding_layer(p_out9,m_out9,is_training,(p_out6.get_shape().as_list()[1],p_out6.get_shape().as_list()[2]),
                                        p_out6,m_out6,filter_numbers1=32)

    with tf.variable_scope("decoding10") as scope:
        p_out11,m_out11 = decoding_layer(p_out10,m_out10,is_training,(p_out5.get_shape().as_list()[1],p_out5.get_shape().as_list()[2]),
                                        p_out5,m_out5,filter_numbers1=16)

    with tf.variable_scope("decoding11") as scope:
        p_out12,m_out12 = decoding_layer(p_out11,m_out11,is_training,(p_out4.get_shape().as_list()[1],p_out4.get_shape().as_list()[2]),
                                        p_out4,m_out4,filter_numbers1=16)

    with tf.variable_scope("decoding12") as scope:
        p_out13,m_out13 = decoding_layer(p_out12,m_out12,is_training,(p_out3.get_shape().as_list()[1],p_out3.get_shape().as_list()[2]),
                                        p_out3,m_out3,filter_numbers1=8)

    with tf.variable_scope("decoding13") as scope:
        p_out14,m_out14 = decoding_layer(p_out13,m_out13,is_training,(p_out2.get_shape().as_list()[1],p_out2.get_shape().as_list()[2]),
                                        p_out2,m_out2,filter_numbers1=4)

    with tf.variable_scope("decoding14") as scope:
        near_pixel1,near_mask1 = near_up_sampling(p_out14,m_out14,(pixel.get_shape().as_list()[1],pixel.get_shape().as_list()[2]))
        pixel_hole = tf.multiply(pixel, mask, name="multiply_mask")
        concat_pixel,concat_mask = concat(near_pixel1, pixel_hole, near_mask1, mask)
        pixel_out,mask_out = partial_conv(concat_pixel,concat_mask,is_training,[1,1],filter_numbers=1,stride=[1,1,1,1],
                                        batch_n=False,nonlinearity="none",trans="one")

    return pixel_out,mask_out



def compute_cost(pixel_gt,mask_gt,pixel_pre,hole_pera,valid_pera):
    with tf.name_scope("cost") as scope:
        loss_valid = tf.losses.absolute_difference(tf.multiply(pixel_gt,mask_gt),tf.multiply(pixel_pre,mask_gt), weights=1.0,
                                                   reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        loss_hole = tf.losses.absolute_difference(tf.multiply(pixel_gt,(1-mask_gt)),tf.multiply(pixel_pre,(1-mask_gt)), weights=1.0,
                                                    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        total_loss = (valid_pera*loss_valid + hole_pera*loss_hole)/(hole_pera+valid_pera)

        tf.summary.scalar('loss',total_loss)

        return total_loss


def model(learning_rate,num_epochs,mini_size,break_t,break_v,pt_out,hole_pera,valid_pera,decay_s,decay_rate,
         fil_num):
    #ops.reset_default_graph()
    tf.summary.scalar('learning_rate',learning_rate)
    tf.summary.scalar('batch_size',mini_size)
    tf.summary.scalar('training_break',break_t)
    tf.summary.scalar('validation_break',break_v)
    tf.summary.scalar('print_interval',pt_out)
    tf.summary.scalar('hole_loss_weight',hole_pera)
    tf.summary.scalar('valid_loss_weight',valid_pera)
    tf.summary.scalar('decay_steps',decay_s)
    tf.summary.scalar('decay_rate',decay_rate)
    tf.summary.scalar('max_filter_number',fil_num)

    m = 19488
    #m = 8
    #h = 512
    #w = 512
    #c = 1

    m_val_size = 1888

    #filenames = "/media/antor/Files/ML/Papers/train_mfix.tfrecords"
    filenames = tf.placeholder(tf.string)
    is_training = tf.placeholder(tf.bool)


    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(3000)
    dataset = dataset.batch(mini_size)
    iterator = dataset.make_initializable_iterator(shared_name="iter")
    #tf.add_to_collection('iterator', iterator)
    pix_gt, mask_in = iterator.get_next()

    pix_gt = tf.reshape(pix_gt,[mini_size,256,256,1])
    mask_in = tf.reshape(mask_in,[mini_size,256,256,1])

    tf.summary.image("input_Y",pix_gt,3)
    tf.summary.image("input_M",mask_in,3)

    pixel_out, mask_out = forward_prop(is_training=is_training,pixel=pix_gt, mask=mask_in)


    tf.summary.image("output_Y",pixel_out,3)
    tf.summary.image("output_M",mask_out,3)

    cost = compute_cost(pixel_gt=pix_gt, mask_gt=mask_in, pixel_pre=pixel_out, hole_pera=hole_pera,valid_pera=valid_pera)

    #global_step = tf.Variable(0, trainable=False)
    #learning_rate_d = tf.train.exponential_decay(learning_rate, global_step,decay_s,decay_rate, staircase=False)
    #tf.summary.scalar('learning_rate_de',learning_rate_d)

    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_d,name="adam").minimize(cost,global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name="adam").minimize(cost)

    num_mini = int(m/mini_size)          #must keep this fully divided and num_mini output as int pretty sure it doesn't need
                                    #to be an int
    merge_sum = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())   #for tensorboard

    saver = tf.train.Saver()    #for model saving
    #builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel/')

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    sess.run(init)
    #saver.restore(sess,('/media/antor/Files/main_projects/gitlab/Landsat7_image_inpainting/tf_models/run-20181009225849/my_model.ckpt'))

    sess.run(iterator.initializer,feed_dict={filenames:"/media/antor/Files/ML/tfrecord/slc_inpainting/train.tfrecords"})

    mini_cost = 0.0
    counter = 1
    epoch_cost = 0.0
    epoch = 0


    while True:
        try:

            _ , temp_cost = sess.run([optimizer,cost], feed_dict={is_training:True})

            #mini_cost += temp_cost/num_mini
            mini_cost += temp_cost/pt_out
            epoch_cost += temp_cost/num_mini

            if counter%50 == 0:
                s = sess.run(merge_sum)
                file_writer.add_summary(s,counter)

            if counter%num_mini==0:
                print("cost after epoch " + str(counter/num_mini) + ": " + str(epoch_cost))
                #saver.save(sess,logdir_m+"my_model.ckpt")
                epoch_cost =0.0
                epoch+=1

            #print("cost after epoch " + str(counter/num_mini) + ": " + str(mini_cost))

            #if counter%1==0:
            #    print("mini batch cost of batch " + str(counter) + " is : " + str(temp_cost))

            if counter%pt_out==0:
                print("mini batch cost of batch " + str(counter) + " is : " + str(mini_cost))
                mini_cost =0.0
                #gc.collect()

            #if counter*mini_size>=break_t:
            #    break

            if epoch ==  num_epochs:
                break

            counter = counter + 1
        except tf.errors.OutOfRangeError:
            break

               #for tensorboard

    num_mini_val = int(m_val_size/mini_size)

    counter_val = 1

    sess.run(iterator.initializer,feed_dict={filenames:"/media/antor/Files/ML/tfrecord/slc_inpainting/val.tfrecords"})
    #sess.run(iterator_val.initializer,feed_dict={filenames_val:"/media/antor/Files/ML/Papers/val_mfix.tfrecords"})

    mini_cost_val = 0.0
    epoch_cost_val = 0.0
    mini_br = int(break_v/mini_size)


    while True:
        try:

            #temp_cost_val = sess.run(cost, feed_dict={M:mask_in_val,Y:label_in_val,is_training:False})
            temp_cost_val = sess.run(cost, feed_dict={is_training:False})
            #temp_cost_val = sess.run(cost_val)

            epoch_cost_val += temp_cost_val/num_mini_val
            mini_cost_val +=  temp_cost_val/mini_br

            if counter_val==num_mini_val:
                print("cost after epoch : " + str(counter_val) +"  " +str(epoch_cost_val))
                #s = sess.run(merge_sum)
                #file_writer.add_summary(s,counter_val)
                #final_val += epoch_cost_val/
                epoch_cost_val =0.0

            #if counter_val*mini_size>=break_v:
            #    print("cost of val set : " + str(mini_cost_val))
            #    s = sess.run(merge_sum)
            #    file_writer.add_summary(s,counter_val)
            #    break


            counter_val = counter_val + 1
        except tf.errors.OutOfRangeError:
            #print(final_val)
            break
    file_writer.close()

    #now_m = datetime.utcnow().strftime("%Y%m%d%H%M%S")    #for tensorboard
    #root_logdir_m = "tf_models"
    #logdir_m = "{}/run-{}/".format(root_logdir_m, now_m)
    #saver.save(sess,logdir_m+"my_model.ckpt")

    sess.close()
