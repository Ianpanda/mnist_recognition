# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:06:10 2017

@author: Lucy Jia
"""

__author__ = 'Jia Chen'

import tensorflow as tf
import numpy as np
import time
import cv2
from feature import histogram,contour,correct,sharpening

def feature_select(data, data_num, choose):
    if choose == 'histogram':
        data_new = histogram.histogram(data, data_num)
    elif choose == 'contour':
        data_new = contour.contour(data, data_num)
    elif choose == 'correct':
        data_new = correct.correct(data, data_num)
    elif choose == 'sharpening':
        data_new = sharpening.sharpening(data, data_num)
    elif choose == 'original':
        data_new = data

    return data_new

def name_select(name_ori):
    if name_ori == 'histogram':
        name_new = 'HOG特征'
    elif name_ori == 'contour':
        name_new = '轮廓特征'
    elif name_ori == 'correct':
        name_new = '偏移校正'
    elif name_ori == 'sharpening':
        name_new = '图像锐化'
    elif name_ori == 'original':
        name_new = '原始图像'
    return name_new

def image_resize(data, data_num, ori_size = 18, new_size = 28):
    image_tensor = []
    for i in range(data_num):
        image_show_ori = data[i].reshape(ori_size, ori_size)
        image_show_ori = cv2.resize(image_show_ori, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        image_show_ori = image_show_ori.flatten()
        image_tensor.append(image_show_ori)
    data_new = np.array(image_tensor)
    data_new = data_new.reshape(data_num, (new_size * new_size))
    return data_new

def CNN(train_data, test_data, times = 200, batch_size = 50, feature_dim = 784, feature_name = 'original'):

    def conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    test_x = test_data[0]
    test_y = test_data[1]
    learning_rate = 0.01
    image_dim = int(np.sqrt(feature_dim))
    # hog image_size 18*18 to 28*28
    if feature_name == 'histogram':
        test_x = image_resize(test_x, test_x.shape[0])
    kernel_size = 5
    conv1_outdim = 32
    conv2_outdim = 64
    features_dim = 1024
    
    # write logs
    filedir = r'./tmp/mnist_cnn_logs/'
    filename = filedir + feature_name

    sess = tf.InteractiveSession()

    # model
    with tf.name_scope('input_layer'):
        x = tf.placeholder("float", shape=[None, feature_dim], name='image_vectors')
        x_images = tf.reshape(x,[-1,image_dim,image_dim,1])
        tf.summary.image('images_input',x_images,6)
        y_true = tf.placeholder("float", shape=[None, 10], name='labels')

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([kernel_size, kernel_size, 1, conv1_outdim])
        b_conv1 = bias_variable([conv1_outdim])
        h_conv1 = tf.nn.relu(conv2d(x_images, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)
 
    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([kernel_size, kernel_size, conv1_outdim, conv2_outdim])
        b_conv2 = bias_variable([conv2_outdim])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        image_newdim = image_dim // 4
        W_fc1 = weight_variable([image_newdim * image_newdim * conv2_outdim, features_dim])
        b_fc1 = bias_variable([features_dim])

        h_pool2_flat = tf.reshape(h_pool2, [-1, image_newdim * image_newdim * conv2_outdim])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([features_dim, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='predict')

    if feature_name in ['original', 'histogram']:
        tf.summary.histogram('weights_conv1',W_conv1)
        tf.summary.histogram('bias_conv1', b_conv1)
        tf.summary.histogram('weights_fc2', W_fc2)
        tf.summary.histogram('bias_fc2', b_fc2)
        tf.summary.histogram('predict', y_conv)

    # accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy',accuracy)

    # traning
    with tf.name_scope('train'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_conv + 1e-10), reduction_indices=1))
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        tf.summary.scalar('loss_function', cross_entropy)

    tf.global_variables_initializer().run()
    merged_op = tf.summary.merge_all()
    if tf.gfile.Exists(filename):
        tf.gfile.DeleteRecursively(filename)
    summary_writer = tf.summary.FileWriter(filename, sess.graph)
    start = time.clock()
    for j in range(times):
        batch = train_data.train.next_batch(batch_size)
        batch = feature_select(batch, batch_size, feature_name)
        train_x = batch[0]
        train_y = batch[1]
        # hog image_size 18*18 to 28*28
        if feature_name == 'histogram':
            train_x = image_resize(train_x, train_x.shape[0])
        sess.run([train_step, cross_entropy], feed_dict={x: train_x, y_true: train_y, keep_prob: 1.0})
        if j % 10 == 0:
            summary_str = sess.run(merged_op,feed_dict={x: train_x, y_true: train_y, keep_prob: 1.0})
            summary_writer.add_summary(summary_str,j)
            if j % 50 == 0:
                print("训练过程第%d步" % j, end='\t')
                print('准确率为%.2f' % accuracy.eval(feed_dict={x: train_x, y_true: train_y, keep_prob: 1.0}))
    end = time.clock()
    train_time = end - start
    print('最终模型准确率为%.2f' % (accuracy.eval(feed_dict={x: test_x, y_true: test_y, keep_prob: 1.0}) * 100) + r'%')
    print('最终训练时间消耗为%.3f秒' % train_time)
    sess.close()

    # write .bat
    feature_name_new = name_select(feature_name)
    f = open(filedir + feature_name_new + '.bat', 'w')
    f.write('tensorboard --logdir=' + feature_name)
    f.close()
    summary_writer.close()
    Flags = True

    return (Flags, accuracy, train_time)

if __name__ == "__main__":
    import dataload.load_MNIST_data as MNIST
    from feature import histogram,contour,correct,sharpening

    mnist = MNIST.load_MNIST()
    test_num = 100
    train_data = mnist
    test_data = mnist.test.next_batch(test_num)
    # choose_list = ['original','histogram','contour','correct','sharpening']
    choose_list = ['original']
    for choose in choose_list:
        if choose == 'histogram':
            train = train_data
            test = feature_select(test_data, test_num, choose)
            result = CNN(train, test, feature_name=choose)
        elif choose == 'contour':
            train = train_data
            test = feature_select(test_data, test_num, choose)
            result = CNN(train, test, feature_name=choose)
        elif choose == 'correct':
            train = train_data
            test = feature_select(test_data, test_num, choose)
            result = CNN(train, test, feature_name=choose)
        elif choose == 'sharpening':
            train = train_data
            test = feature_select(test_data, test_num, choose)
            result = CNN(train, test, feature_name=choose)
        elif choose == 'original':
            train = train_data
            test = mnist.test.next_batch(test_num)
            result = CNN(train, test, feature_name=choose)

    if result != None:
        print('CNN分类完毕！')
    else:
        print('CNN运行错误，请检查参数设置！')