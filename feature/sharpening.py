# coding:utf-8 #
__author__ = 'Bai Bo'

import numpy as np
import cv2

# from PIL import Image
# import matplotlib.pyplot as plt

def image_sharpening(image):
    image = cv2.medianBlur(image,3)
    image_re = cv2.Laplacian(image,-1,ksize=1)
    image = image - image_re

    return image

def sharpening(data, number):
    image_tensor = []
    train_x_ori = data[0]
    train_y = data[1]
    for each_image_ori in train_x_ori:
        each_image = np.copy(each_image_ori)
        each_image *= 255
        each_image.shape = (28, 28)
        each_image = image_sharpening(each_image)
        each_image = each_image.flatten()
        image_tensor.append(each_image)
    train_x = np.array(image_tensor)
    train_x = train_x.reshape(number, 784)
    # image_ori = Image.fromarray(train_x_ori[0].reshape(28, 28))
    # image = Image.fromarray(train_x[0].reshape(28, 28))
    # plt.figure()
    # plt.imshow(image_ori)
    # plt.figure()
    # plt.imshow(image)
    # plt.show()

    return (train_x, train_y)

if __name__ == '__main__':
    import dataload.load_MNIST_data as MNIST
    mnist = MNIST.load_MNIST()
    train_data = mnist.train.next_batch(50)
    train = sharpening(train_data, 50)
    if train != None:
        print('图像滤波特征提取成功！')
