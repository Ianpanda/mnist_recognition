__author__ = 'Bai Bo'
import tensorflow as tf
import numpy as np
import time
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

def Softmax(train_data, test_data, times = 1000, batch_size = 50, feature_dim = 784, feature_name = 'original'):
    test_x = test_data[0]
    test_y = test_data[1]
    learning_rate = 0.01
    image_dim = int(np.sqrt(feature_dim))
    # write logs
    filedir = r'./tmp/mnist_softmax_logs/'
    filename = filedir + feature_name

    sess = tf.InteractiveSession()
    # model
    with tf.name_scope('input_layer'):
        x = tf.placeholder("float", shape=[None, feature_dim], name='image_vectors')
        x_images = tf.reshape(x,[-1,image_dim,image_dim,1])
        tf.summary.image('images_input',x_images,6)
        y_true = tf.placeholder("float", shape=[None, 10], name='labels')
    with tf.name_scope('hidden_layer'):
        W = tf.Variable(tf.zeros([feature_dim, 10]), name='weights')
        b = tf.Variable(tf.zeros([10]), name='bias')
        y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict')
        if feature_name in ['original','histogram']:
            tf.summary.histogram('weights', W)
            tf.summary.histogram('bias', b)
            tf.summary.histogram('predict', y)

    # accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy',accuracy)

    # traning
    with tf.name_scope('train'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y + 1e-10), reduction_indices=1))
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
        sess.run([train_step, cross_entropy], feed_dict={x: train_x, y_true: train_y})
        if j % 10 == 0:
            summary_str = sess.run(merged_op,feed_dict={x: train_x, y_true: train_y})
            summary_writer.add_summary(summary_str,j)
        if j % 50 == 0:
            print("训练过程第%d步" % j, end='\t')
            print('准确率为%.2f' % accuracy.eval(feed_dict={x: train_x, y_true: train_y}))
    end = time.clock()
    train_time = end - start
    print('最终模型准确率为%.2f' % (accuracy.eval(feed_dict={x: test_x, y_true: test_y}) * 100) + r'%')
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
            result = Softmax(train, test, feature_dim=324, feature_name=choose)
        elif choose == 'contour':
            train = train_data
            test = feature_select(test_data, test_num, choose)
            result = Softmax(train, test, feature_dim=784, feature_name=choose)
        elif choose == 'correct':
            train = train_data
            test = feature_select(test_data, test_num, choose)
            result = Softmax(train, test, feature_dim=784, feature_name=choose)
        elif choose == 'sharpening':
            train = train_data
            test = feature_select(test_data, test_num, choose)
            result = Softmax(train, test, feature_dim=784, feature_name=choose)
        elif choose == 'original':
            train = train_data
            test = mnist.test.next_batch(test_num)
            result = Softmax(train, test, feature_dim=784, feature_name=choose)

    if result != None:
        print('Softmax回归分类完毕！')
    else:
        print('Softmax回归运行错误，请检查参数设置！')