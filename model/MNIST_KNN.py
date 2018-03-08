__author__ = 'Bai Bo'
import tensorflow as tf
import numpy as np
import time

def name_select(name_ori):
    if name_ori == 'histogram':
        name_new = 'HOG特征'
    elif name_ori == 'contour':
        name_new = '轮廓特征'
    elif name_ori == 'correct':
        name_new = '偏移校正'
    elif name_ori == 'sharpening':
        name_new = '图像锐化'
    elif name_ori == 'subsampling':
        name_new = '池化特征'
    elif name_ori == 'original':
        name_new = '原始图像'
    return name_new

def KNN(train_data, test_data, feature_dim = 784, feature_name = 'original', type = 1):
    train_x = train_data[0]
    train_y = train_data[1]
    test_x = test_data[0]
    test_y = test_data[1]
    distance = 0
    image_dim = int(np.sqrt(feature_dim))
    # write logs
    filedir = r'./tmp/mnist_knn_logs/'
    filename = filedir + feature_name

    sess = tf.InteractiveSession()
    with tf.name_scope('input_layer'):
        x_train = tf.placeholder(tf.float32,[None,feature_dim],name='train_image_vectors')
        x_images = tf.reshape(x_train, [-1,image_dim,image_dim,1])
        x_test = tf.placeholder(tf.float32,[feature_dim],name='test_image_vectors')
        tf.summary.image('images_input', x_images, 6)
    if type == 1:
        distance = tf.reduce_sum(tf.abs(tf.add(x_train, tf.negative(x_test))), reduction_indices=1, name='L1_distance') #L1距离
        Flags = True
    elif type == 2:
        distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(x_train,tf.negative(x_test)),2),reduction_indices=1), name='L2_distance') #L2距离
        Flags = True
    else:
        Flags = False

    pred = tf.argmin(distance, 0, name='predict_labels')

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    if tf.gfile.Exists(filename):
        tf.gfile.DeleteRecursively(filename)
    summary_writer = tf.summary.FileWriter(filename, sess.graph)
    start = time.clock()
    right = 0
    # label_pred = []
    # label_true = []
    for i in range(test_x.shape[0]):
        ansIndex = sess.run(pred, feed_dict={x_train:train_x, x_test:test_x[i, :]})
        # label_pred.append(np.argmax(train_y[ansIndex]))
        # label_true.append(np.argmax(test_y[i]))
        print('预测结果:', np.argmax(train_y[ansIndex]), end='\t')
        print('真实结果:', np.argmax(test_y[i]))
        if np.argmax(test_y[i]) == np.argmax(train_y[ansIndex]):
            right += 1.0
        accuracy_trend = right/(test_x.shape[0])
        summary_accuracy = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=accuracy_trend)])
        summary_writer.add_summary(summary_accuracy, i)
    merged_op = tf.summary.merge_all()
    for i in range(test_x.shape[0]):
        if i % 50 == 0:
            summary_str = sess.run(merged_op, feed_dict={x_train: train_x, x_test: test_x[i, :]})
            summary_writer.add_summary(summary_str,  i)
    accuracy = right/(test_x.shape[0])
    end = time.clock()
    train_time = end - start
    print ('最终模型准确率为%.2f' % (accuracy*100) + r'%')
    print('最终训练时间消耗为%.3f秒' % train_time)

    # write .bat
    feature_name_new = name_select(feature_name)
    f = open(filedir + feature_name_new + '.bat', 'w')
    f.write('tensorboard --logdir=' + feature_name)
    f.close()
    sess.close()
    summary_writer.close()

    return (Flags, accuracy)

if __name__ == "__main__":
    import dataload.load_MNIST_data as MNIST
    # import feature as extract
    from feature import histogram,contour,correct,sharpening,subsampling

    mnist = MNIST.load_MNIST()
    train_num = 1000
    test_num = 500
    train_data = mnist.train.next_batch(train_num)
    test_data = mnist.test.next_batch(test_num)
    # choose_list = ['correct','original','histogram','contour','subsampling','sharpening']
    choose_list = ['original']
    for choose in choose_list:
        if choose == 'histogram':
            train = histogram.histogram(train_data, train_num)
            test = histogram.histogram(test_data, test_num)
            result = KNN(train, test, feature_dim=324, feature_name=choose, type=1)
        elif choose == 'subsampling':
            train = subsampling.subsampling(train_data, train_num)
            test = subsampling.subsampling(test_data, test_num)
            result = KNN(train, test, feature_dim=256, feature_name=choose, type=2)
        elif choose == 'contour':
            train = contour.contour(train_data, train_num)
            test = contour.contour(test_data, test_num)
            result = KNN(train, test, feature_dim=784, feature_name=choose, type=2)
        elif choose == 'correct':
            train = correct.correct(train_data, train_num)
            test = correct.correct(test_data, test_num)
            result = KNN(train, test, feature_dim=784, feature_name=choose, type=1)
        elif choose == 'sharpening':
            train = sharpening.sharpening(train_data, train_num)
            test = sharpening.sharpening(test_data, test_num)
            result = KNN(train, test, feature_dim=784, feature_name=choose, type=2)
        elif choose == 'original':
            train = train_data
            test = test_data
            result = KNN(train, test, feature_dim=784, feature_name=choose, type=1)

    if result[0] == True:
        print('KNN分类完毕！')
    else:
        print('KNN运行错误，请检查参数设置！')