# coding:utf-8 #
__author__ = 'Bai Bo'

import tensorflow as tf
import numpy as np
from PIL import Image

# import matplotlib.pyplot as plt

def resize_image(image, width, height):
    return image.resize((width, height), Image.ANTIALIAS)

def max_pooling(image):
    return tf.nn.max_pool(image, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


def image2tensor(image):
    image = np.array(resize_image(image, 32, 32))
    image = tf.expand_dims(image, -1)
    return image

def tensor2array(name):
    sess = tf.InteractiveSession()
    array_re = name.eval()
    return array_re

def subsampling(data, number):
    image_tensor = []
    train_x_ori = data[0]
    train_y = data[1]
    for each_image_ori in train_x_ori:
        each_image = np.copy(each_image_ori)
        each_image.shape = (28,28)
        each_image = Image.fromarray(each_image)
        each_image = image2tensor(each_image)
        each_image = tf.reshape(each_image, [-1])
        image_tensor.append(each_image)
    image_tensor = tf.reshape(image_tensor, [number,32,32,1])
    image_pooling = max_pooling(image_tensor)
    train_x = tf.reshape(image_pooling, [-1,256])
    train_x = tensor2array(train_x)

    # image = Image.fromarray(train_x[0].reshape(16, 16) * 255)
    # plt.figure()
    # plt.imshow(image)
    # plt.show()

    return (train_x, train_y)

if __name__ == '__main__':
    import dataload.load_MNIST_data as MNIST
    mnist = MNIST.load_MNIST()
    train_data = mnist.train.next_batch(5000)
    train = subsampling(train_data, 5000)
    if train != None:
        print('池化特征提取成功！')
