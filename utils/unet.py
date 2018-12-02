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
            out_shape_list[1] = pixel.get_shape().as_list()[1] + 2
            out_shape_list[2] = pixel.get_shape().as_list()[2] + 2
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
            #up_pixel = tf.nn.sigmoid(normalized_out, name="sigmoid")
            up_pixel = tf.nn.tanh(normalized_out, name="tanh")

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", B)
        tf.summary.histogram("activations", up_pixel)

        return up_pixel


def near_up_sampling(pixel, output_size):
    with tf.name_scope("nearest_up") as scope:
        up_pixel = tf.image.resize_nearest_neighbor(pixel, size=output_size, name="nearest_pixel_up")
        return up_pixel

def concat(near_pixel, pconv_pixel, near_mask, pconv_mask):
    with tf.name_scope("concatenation") as scope:
        up_pixel = tf.concat([pconv_pixel, near_pixel], axis=3)
        up_mask = tf.concat([pconv_mask,near_mask], axis=3)
        return up_pixel, up_mask

def decoding_layer(pixel_in, output_size_in, filter_numbers1):
    with tf.name_scope("decoding") as scope:
        near_pixel1 = near_up_sampling(pixel_in,output_size_in)
        #concat_pixel,concat_mask = concat(near_pixel1, pconv_pixel1, near_mask1, pconv_mask1)
        pixel_out = conv_block(near_pixel1,[3,3],filter_numbers1,[1,1,1,1],"leaky_relu",conv_t="dconv")

        return pixel_out

def forward_prop( pixel):
    non_lin = "relu"

    with tf.variable_scope("PConv1") as scope:
        p_out2 = conv_block(pixel,kernel_size=[3,3],filter_numbers=4,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="conv")

    with tf.variable_scope("PConv2") as scope:
        p_out3 = conv_block(p_out2,kernel_size=[3,3],filter_numbers=8,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="conv")

    with tf.variable_scope("PConv3") as scope:
        p_out4 = conv_block(p_out3,kernel_size=[3,3],filter_numbers=16,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="conv")

    with tf.variable_scope("PConv4") as scope:
        p_out5 = conv_block(p_out4,kernel_size=[3,3],filter_numbers=16,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="conv")

    with tf.variable_scope("PConv5") as scope:
        p_out6 = conv_block(p_out5,kernel_size=[3,3],filter_numbers=32,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="conv")

    with tf.variable_scope("PConv6") as scope:
        p_out7 = conv_block(p_out6,kernel_size=[3,3],filter_numbers=32,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="conv")

    with tf.variable_scope("PConv7") as scope:
        p_out8 =  conv_block(p_out7,kernel_size=[3,3],filter_numbers=32,stride=[1,1,1,1],nonlinearity=non_lin,conv_t="conv")

    with tf.variable_scope("decoding8") as scope:
        p_out9 = decoding_layer(p_out8,(p_out7.get_shape().as_list()[1],p_out7.get_shape().as_list()[2]),filter_numbers1=32)

    with tf.variable_scope("decoding9") as scope:
        p_out10 = decoding_layer(p_out9,(p_out6.get_shape().as_list()[1],p_out6.get_shape().as_list()[2]),filter_numbers1=32)

    with tf.variable_scope("decoding10") as scope:
        p_out11 = decoding_layer(p_out10,(p_out5.get_shape().as_list()[1],p_out5.get_shape().as_list()[2]),filter_numbers1=16)

    with tf.variable_scope("decoding11") as scope:
        p_out12 = decoding_layer(p_out11,(p_out4.get_shape().as_list()[1],p_out4.get_shape().as_list()[2]),filter_numbers1=16)

    with tf.variable_scope("decoding12") as scope:
        p_out13 = decoding_layer(p_out12,(p_out3.get_shape().as_list()[1],p_out3.get_shape().as_list()[2]),filter_numbers1=8)

    with tf.variable_scope("decoding13") as scope:
        p_out14 = decoding_layer(p_out13,(p_out2.get_shape().as_list()[1],p_out2.get_shape().as_list()[2]),filter_numbers1=4)

    with tf.variable_scope("decoding14") as scope:
        near_pixel1 = near_up_sampling(p_out14,(pixel.get_shape().as_list()[1],pixel.get_shape().as_list()[2]))
        #pixel_hole = tf.multiply(pixel, mask, name="multiply_mask")
        #concat_pixel,concat_mask = concat(near_pixel1, pixel_hole, near_mask1, mask)
        pixel_out = conv_block(near_pixel1,kernel_size=[1,1],filter_numbers=1,stride=[1,1,1,1],nonlinearity="none",conv_t="conv")


    return pixel_out,p_out3,p_out5,p_out7



def compute_cost(pixel_gt,pixel_pre):
    with tf.name_scope("cost") as scope:
        loss_valid = tf.losses.absolute_difference(pixel_gt,pixel_pre, weights=1.0,reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        # loss_hole = tf.losses.absolute_difference(tf.multiply(pixel_gt,(1-mask_gt)),tf.multiply(pixel_pre,(1-mask_gt)), weights=1.0,
        #                                             reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        #total_loss = (valid_pera*loss_valid + hole_pera*loss_hole)/(hole_pera+valid_pera)
        #total_loss = (valid_pera*loss_valid + hole_pera*loss_hole)/(hole_pera+valid_pera)

        tf.summary.scalar('loss',loss_valid)

        return loss_valid


def model(learning_rate,num_epochs,mini_size,break_t,break_v,pt_out,hole_pera,valid_pera,decay_s,decay_rate,
         fil_num):
    #ops.reset_default_graph()
    tf.summary.scalar('learning_rate',learning_rate)
    tf.summary.scalar('batch_size',mini_size)
    tf.summary.scalar('training_break',break_t)
    tf.summary.scalar('validation_break',break_v)
    tf.summary.scalar('print_interval',pt_out)
    #tf.summary.scalar('hole_loss_weight',hole_pera)
    #tf.summary.scalar('valid_loss_weight',valid_pera)
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
    #filenames = tf.placeholder(tf.string)
    #is_training = tf.placeholder(tf.bool)
    dataset = tf.data.TFRecordDataset("/media/antor/Files/ML/tfrecord/slc_inpainting/train_unet.tfrecords")
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(3000)
    dataset = dataset.batch(mini_size)
    dataset = dataset.repeat(num_epochs)

    dataset_v = tf.data.TFRecordDataset("/media/antor/Files/ML/tfrecord/slc_inpainting/val_unet.tfrecords")
    dataset_v = dataset_v.map(_parse_function)
    dataset_v = dataset_v.batch(mini_size)
    dataset_v = dataset_v.repeat(num_epochs)
    #iterator = dataset.make_initializable_iterator(shared_name="iter")
    #tf.add_to_collection('iterator', iterator)
    handle = tf.placeholder(tf.string)

    iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types)

    pix_gt1 = iterator.get_next()

    training_iterator = dataset.make_initializable_iterator()
    validation_iterator = dataset_v.make_initializable_iterator()

    pix_gt = tf.reshape(pix_gt1,[mini_size,256,256,1])

    tf.summary.image("input_Y",pix_gt,3)

    pixel_out,out1,out2,out3 = forward_prop(pixel=pix_gt)

    tf.summary.image("output_Y",pixel_out,3)

    cost = compute_cost(pixel_gt=pix_gt,pixel_pre=pixel_out)

    #cost_v = tf.summary.scalar("loss_val", epoch_cost_v)

    #cost_v = tf.summary.scalar('val_loss',cost)
    #global_step = tf.Variable(0, trainable=False)
    #learning_rate_d = tf.train.exponential_decay(learning_rate, global_step,decay_s,decay_rate, staircase=False)
    #tf.summary.scalar('learning_rate_de',learning_rate_d)

    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_d,name="adam").minimize(cost,global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name="adam").minimize(cost)

    num_mini = int(m/mini_size)          #must keep this fully divided and num_mini output as int pretty sure it doesn't need
                                    #to be an int
    #merge_sum = tf.summary.merge_all()
    #merge_sum = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())    #for tensorboard
    file_writer_v = tf.summary.FileWriter(logdir_v, tf.get_default_graph())    #for tensorboard

    #saver = tf.train.Saver()    #for model saving
    #builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel/')

    init = tf.global_variables_initializer()

    sess = tf.Session()

    sess.run(init)
    #saver.restore(sess,('/media/antor/Files/main_projects/gitlab/Landsat7_image_inpainting/tf_models/run-20181009225849/my_model.ckpt'))

    #sess.run(iterator.initializer,feed_dict={filenames:"/media/antor/Files/ML/tfrecord/slc_inpainting/train.tfrecords"})
    sess.run(training_iterator.initializer)
    sess.run(validation_iterator.initializer)
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    mini_cost = 0.0
    counter = 1
    epoch_cost = 0.0
    epoch = 1
    num_mini_val = int(m_val_size/mini_size)
    while True:
        try:
            _ , temp_cost = sess.run([optimizer,cost], feed_dict={handle : training_handle})

            #mini_cost += temp_cost/num_mini
            mini_cost += temp_cost/pt_out
            epoch_cost += temp_cost/num_mini

            if counter%50 == 0:
                merge_sum = tf.summary.merge_all()
                #merge_sum = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, train_scope))
                s = sess.run(merge_sum,feed_dict={handle : training_handle})
                file_writer.add_summary(s,counter)
            if counter%pt_out==0:
                print("mini batch cost of batch " + str(counter) + " is : " + str(mini_cost))
                mini_cost =0.0


            if counter%num_mini==0:
                print("cost after epoch " + str(counter/num_mini) + ": " + str(epoch_cost))

                #saver.save(sess,logdir_m+"my_model.ckpt")
                # s = sess.run(merge_sum)
                # file_writer.add_summary(s,counter)
                #file_writer_t.add_summary(s_t,counter)
                #file_writer_t.flush()
                counter_v    = 1
                epoch_cost_v = 0.0
                while True:
                    try:
                        #if counter_v%36==0:
                        #    tf.get_default_graph().clear_collection('total_accuracy')
                        temp_cost_v = sess.run(cost,feed_dict={handle : validation_handle})
                        epoch_cost_v += temp_cost_v/num_mini_val



                        if counter_v%num_mini_val==0:
                            print("val cost at epoch  " + str(epoch) + ": " + str(epoch_cost_v))
                            merge_sum_v = tf.summary.merge_all()
                            #merge_sum = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, train_scope))
                            s_v = sess.run(merge_sum_v,feed_dict={handle : validation_handle})
                            file_writer_v.add_summary(s_v,counter)

                            # with tf.name_scope("cost_val_epoch") as scope:
                            #     cost_v = tf.summary.scalar("loss_val", epoch_cost_v)
                            #
                            # merge_val = tf.summary.merge([cost_v])
                            # s_v = sess.run(merge_val,feed_dict={handle : validation_handle})
                            # file_writer.add_summary(s_v,counter)

                            epoch_cost_v = 0.0
                            #s_v = sess.run(merge_sum, feed_dict={handle : validation_handle, decision:False})
                            #file_writer_v.add_summary(s_v,counter)
                            #file_writer_v.flush()
                            #s = sess.run(merge_sum)
                            #file_writer.add_summary(s,counter)

                            break
                        counter_v+=1
                    except tf.errors.OutOfRangeError:
                        #tf.summary.scalar('dev_epoch_cost',epoch_cost_v)
                        break
                epoch_cost =0.0
                epoch+=1

            #print("cost after epoch " + str(counter/num_mini) + ": " + str(mini_cost))

            #if counter%1==0:
            #    print("mini batch cost of batch " + str(counter) + " is : " + str(temp_cost))


                #gc.collect()

            #if counter*mini_size>=break_t:
            #    break

            #if epoch ==  num_epochs:
                #break

            counter = counter + 1
        except tf.errors.OutOfRangeError:
            builder = tf.saved_model.builder.SavedModelBuilder(logdir_m)
            inputs = {'input_x': tf.saved_model.utils.build_tensor_info(pix_gt)}
            outputs = {'output1' : tf.saved_model.utils.build_tensor_info(out1),
                        'output2' : tf.saved_model.utils.build_tensor_info(out2),
                        'output3' : tf.saved_model.utils.build_tensor_info(out3)}
            signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'test_sig_name')
            builder.add_meta_graph_and_variables(sess, ['test_saved_model'], {'test_signature':signature})
            builder.save()

            break

               #for tensorboard


    sess.close()


model(learning_rate=.0005,num_epochs=1,mini_size=16,break_t=7000,break_v=700,pt_out=25,hole_pera=6.0,
     valid_pera=1.0,decay_s=538.3,decay_rate=.96,fil_num=32)

# from tensorflow.python.framework import ops
# f = np.random.uniform(np.log10(.00005),np.log10(.01),6)
# print(f)
# i = 10**f
#
# print(i)
#
# for l in i:    #for k in b:
#     print(l)
#     model(learning_rate=l,num_epochs=1,mini_size=16,break_t=7000,break_v=700,pt_out=25,hole_pera=6.0,
#          valid_pera=1.0,decay_s=538.3,decay_rate=.96,fil_num=32)
#
#     ops.reset_default_graph()
