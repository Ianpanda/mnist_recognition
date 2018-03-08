# coding:utf-8 #
__author__ = 'Bai Bo'

import numpy as np
import cv2

# from PIL import Image
# import matplotlib.pyplot as plt

def get_hog(image,hog):
    # image = image[1]
    image = image.astype(np.uint8)
    hist = hog.compute(image)
    return hist

def histogram(data,number):
    image_tensor = []
    train_x_ori = data[0]
    train_y = data[1]
    winSize = (28,28)
    blockSize = (14,14)
    blockStride = (7,7)
    cellSize = (7,7)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    for each_image_ori in train_x_ori:
        each_image = np.copy(each_image_ori)
        each_image *= 255
        each_image.shape = (28,28)
        each_image_hist = get_hog(each_image,hog)
        each_image_hist = np.transpose(each_image_hist)
        image_tensor.append(each_image_hist)
    train_x = np.array(image_tensor)
    train_x = train_x.reshape(number,-1)

    # image_ori = Image.fromarray(train_x_ori[0].reshape(28, 28))
    # # image = Image.fromarray(train_x)
    # cv_image = train_x[0]
    # cv_image = cv2.cvtColor(cv_image,cv2.COLOR_GRAY2BGR)
    # cv2.namedWindow('HOG',0)
    # cv2.imshow('HOG',cv_image)
    # plt.figure()
    # plt.imshow(image_ori)
    # # plt.figure()
    # # plt.imshow(image)
    # plt.show()

    return (train_x, train_y)

if __name__ == '__main__':
    import dataload.load_MNIST_data as MNIST
    mnist = MNIST.load_MNIST()
    num = 50
    train = mnist.train.next_batch(num)
    train = histogram(train,num)
    if train != None:
        print('图像HOG特征提取成功！')
