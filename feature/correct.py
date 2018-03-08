# coding:utf-8 #
__author__ = 'Bai Bo'

import numpy as np
import cv2
from skimage import morphology

# from PIL import Image
# import matplotlib.pyplot as plt

def image_correct(image):
    image = cv2.threshold(image,50,255,cv2.THRESH_BINARY)
    image = image[1]
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # angle = -angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h,w) = image.shape[:2]
    center = (w // 2,h // 2)
    M = cv2.getRotationMatrix2D(center,angle,1)
    image = cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_REPLICATE)
    image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)), iterations=2)

    return image

def correct(data,number):
    image_tensor = []
    train_x_ori = data[0]
    train_y = data[1]
    for each_image_ori in train_x_ori:
        each_image = np.copy(each_image_ori)
        each_image *= 255
        each_image.shape = (28,28)
        each_image = image_correct(each_image)
        each_image= each_image.flatten()
        image_tensor.append(each_image)
    train_x = np.array(image_tensor)
    train_x = train_x.reshape(number,784)

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
    num = 50
    train = mnist.train.next_batch(num)
    train = correct(train,num)
    if train != None:
        print('图像偏移矫正成功！')
