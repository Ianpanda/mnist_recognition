#! /usr/bin/env python
# coding:utf-8
__author__ = 'Bai Bo and Jia Chen'

import sys
# import numpy as np
# import tensorflow as tf
import cv2

from mnist_gui_design import Ui_MainWindow                        # py文件导入
from PyQt5 import QtGui, QtWidgets

MainWindow=Ui_MainWindow                                         # py文件导入

def feature_extract(data, data_num, choose, image_dim = 28, option = False):
    if choose == 'histogram':
        data_new = histogram.histogram(data, data_num)
    elif choose == 'contour':
        data_new = contour.contour(data, data_num)
    elif choose == 'correct':
        data_new = correct.correct(data, data_num)
    elif choose == 'sharpening':
        data_new = sharpening.sharpening(data, data_num)
    elif choose == 'subsampling':
        data_new = subsampling.subsampling(data, data_num)
    elif choose == 'original':
        data_new = data
    if option == True:
        data = data[0] * 255
        data_new = data_new[0] * 255
        for i in range(4):
            image_show_ori = data[i].reshape(28,28)
            image_show_ori = cv2.resize(image_show_ori, (140,140), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(r'./ori_'+str(i)+'.jpg', image_show_ori)
            image_show = data_new[i].reshape(image_dim, image_dim)
            image_show = cv2.resize(image_show,(140,140), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(r'./feature_'+str(i)+'.jpg',image_show)
    return data_new

class MyApp(QtWidgets.QMainWindow, MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        MainWindow.__init__(self)
        self.setupUi(self)
        self.Combo_feature.activated[str].connect(self.onActivate_feature)
        self.Combo_model.activated[str].connect(self.onActivate_model)
        self.pushButton.clicked.connect(self.Run_model)

    # show the feature result
    def onActivate_feature(self):
        global train_data_show
        choose = self.Combo_feature.currentText()
        if choose == "轮廓特征":
            image_data = feature_extract(train_data_show, 4, 'contour', option=True)
            print(choose+'提取成功！')
        elif choose == "偏移校正":
            image_data = feature_extract(train_data_show, 4, 'correct', option=True)
            print(choose + '提取成功！')
        elif choose == "图像锐化":
            image_data = feature_extract(train_data_show, 4, 'sharpening', option=True)
            print(choose + '提取成功！')
        elif choose == "HOG特征":
            image_data = feature_extract(train_data_show, 4, 'histogram',image_dim = 18, option=True)
            print(choose + '提取成功！')
        elif choose == "池化特征":
            image_data = feature_extract(train_data_show, 4, 'subsampling',image_dim = 16, option=True)
            print(choose + '提取成功！')
        elif choose == "原始图像":
            image_data = feature_extract(train_data_show, 4, 'original', option=True)
            print(choose + '提取成功！')
        pixmap_ori_0 = QtGui.QPixmap()
        pixmap_new_0 = QtGui.QPixmap()
        pixmap_ori_0.load(r'./ori_0.jpg')
        pixmap_new_0.load(r'./feature_0.jpg')
        self.scene_ori_0 = QtWidgets.QGraphicsScene(self)
        self.scene_new_0 = QtWidgets.QGraphicsScene(self)
        self.scene_ori_0.addItem(QtWidgets.QGraphicsPixmapItem(pixmap_ori_0))
        self.scene_new_0.addItem(QtWidgets.QGraphicsPixmapItem(pixmap_new_0))
        self.graphicsView_ori_0.setScene(self.scene_ori_0)
        self.graphicsView_new_0.setScene(self.scene_new_0)
        pixmap_ori_1 = QtGui.QPixmap()
        pixmap_new_1 = QtGui.QPixmap()
        pixmap_ori_1.load(r'./ori_1.jpg')
        pixmap_new_1.load(r'./feature_1.jpg')
        self.scene_ori_1 = QtWidgets.QGraphicsScene(self)
        self.scene_new_1 = QtWidgets.QGraphicsScene(self)
        self.scene_ori_1.addItem(QtWidgets.QGraphicsPixmapItem(pixmap_ori_1))
        self.scene_new_1.addItem(QtWidgets.QGraphicsPixmapItem(pixmap_new_1))
        self.graphicsView_ori_1.setScene(self.scene_ori_1)
        self.graphicsView_new_1.setScene(self.scene_new_1)
        pixmap_ori_2 = QtGui.QPixmap()
        pixmap_new_2 = QtGui.QPixmap()
        pixmap_ori_2.load(r'./ori_2.jpg')
        pixmap_new_2.load(r'./feature_2.jpg')
        self.scene_ori_2 = QtWidgets.QGraphicsScene(self)
        self.scene_new_2 = QtWidgets.QGraphicsScene(self)
        self.scene_ori_2.addItem(QtWidgets.QGraphicsPixmapItem(pixmap_ori_2))
        self.scene_new_2.addItem(QtWidgets.QGraphicsPixmapItem(pixmap_new_2))
        self.graphicsView_ori_2.setScene(self.scene_ori_2)
        self.graphicsView_new_2.setScene(self.scene_new_2)
        pixmap_ori_3 = QtGui.QPixmap()
        pixmap_new_3 = QtGui.QPixmap()
        pixmap_ori_3.load(r'./ori_3.jpg')
        pixmap_new_3.load(r'./feature_3.jpg')
        self.scene_ori_3 = QtWidgets.QGraphicsScene(self)
        self.scene_new_3 = QtWidgets.QGraphicsScene(self)
        self.scene_ori_3.addItem(QtWidgets.QGraphicsPixmapItem(pixmap_ori_3))
        self.scene_new_3.addItem(QtWidgets.QGraphicsPixmapItem(pixmap_new_3))
        self.graphicsView_ori_3.setScene(self.scene_ori_3)
        self.graphicsView_new_3.setScene(self.scene_new_3)

    # change the num train and test by model select
    def onActivate_model(self):
        choose = self.Combo_model.currentText()
        if choose == "Softmax Regression":
            self.textEdit_trainnum.setPlainText('10000')
        elif choose == "CNN":
            self.textEdit_trainnum.setPlainText('30000')
        else:
            self.textEdit_trainnum.setPlainText('1000')

    # determine the model
    def Run_model(self):
        choose_feature = self.Combo_feature.currentText()
        choose_model = self.Combo_model.currentText()
        train_num = int(self.textEdit_trainnum.toPlainText())
        test_num = int(self.textEdit_testnum.toPlainText())
        print('您选择的是%s,，采用%s模型，训练数据%d条，测试数据%d条,已进入训练过程……' % (choose_feature, choose_model, train_num,test_num))
        global mnist
        if choose_model == 'KNN':
            train_data = mnist.train.next_batch(train_num)
            test_data = mnist.test.next_batch(test_num)
            if choose_feature == "HOG特征":
                train = feature_extract(train_data, train_num, 'histogram')
                test = feature_extract(test_data, test_num, 'histogram')
                result = MNIST_KNN.KNN(train, test, feature_dim=324, feature_name='histogram', type=1)
            elif choose_feature == "池化特征":
                train = feature_extract(train_data, train_num, 'subsampling')
                test = feature_extract(test_data, test_num, 'subsampling')
                result = MNIST_KNN.KNN(train, test, feature_dim=256, feature_name='subsampling', type=2)
            elif choose_feature == "轮廓特征":
                train = feature_extract(train_data, train_num, 'contour')
                test = feature_extract(test_data, test_num, 'contour')
                result = MNIST_KNN.KNN(train, test, feature_dim=784, feature_name='contour', type=2)
            elif choose_feature == "偏移校正":
                train = feature_extract(train_data, train_num, 'correct')
                test = feature_extract(test_data, test_num, 'correct')
                result = MNIST_KNN.KNN(train, test, feature_dim=784, feature_name='correct', type=1)
            elif choose_feature == "图像锐化":
                train = feature_extract(train_data, train_num, 'sharpening')
                test = feature_extract(test_data, test_num, 'sharpening')
                result = MNIST_KNN.KNN(train, test, feature_dim=784, feature_name='sharpening', type=2)
            elif choose_feature == "原始图像":
                train = feature_extract(train_data, train_num, 'original')
                test = feature_extract(test_data, test_num, 'original')
                result = MNIST_KNN.KNN(train, test, feature_dim=784, feature_name='original', type=1)

            if result[0] == True:
                print('KNN(最近邻分类器)分类完毕！')
                print('若想要二次构建模型，请退出后重新运行程序，谢谢啦！')
            else:
                print('KNN(最近邻分类器)运行错误，请检查参数设置！')

        elif choose_model == 'Softmax Regression':
            train_num_times = train_num // 50
            test_data = mnist.test.next_batch(test_num)
            if choose_feature == "HOG特征":
                train = mnist
                test = feature_extract(test_data, test_num, 'histogram')
                result = MNIST_Softmax.Softmax(train, test, times=train_num_times, feature_dim=324, feature_name='histogram')
            elif choose_feature == "轮廓特征":
                train = mnist
                test = feature_extract(test_data, test_num, 'contour')
                result = MNIST_Softmax.Softmax(train, test, times=train_num_times, feature_dim=784, feature_name='contour')
            elif choose_feature == "偏移校正":
                train = mnist
                test = feature_extract(test_data, test_num, 'correct')
                result = MNIST_Softmax.Softmax(train, test, times=train_num_times, feature_dim=784, feature_name='correct')
            elif choose_feature == "图像锐化":
                train = mnist
                test = feature_extract(test_data, test_num, 'sharpening')
                result = MNIST_Softmax.Softmax(train, test, times=train_num_times, feature_dim=784, feature_name='sharpening')
            elif choose_feature == "原始图像":
                train = mnist
                test = test_data
                result = MNIST_Softmax.Softmax(train, test, times=train_num_times, feature_dim=784, feature_name='original')
            else:
                result = (False, None)
                print('您选择的特征不适合在此模型上运行，请重新选择！')

            if result[0] == True:
                print('Softmax回归分类完毕！')
                print('若想要二次构建模型，请退出后重新运行程序，谢谢啦！')
            else:
                print('Softmax回归运行错误，请检查参数设置！')

        elif choose_model == 'CNN':
            train_num_times = train_num // 50
            test_data = mnist.test.next_batch(test_num)
            if choose_feature == "HOG特征":
                train = mnist
                test = feature_extract(test_data, test_num, 'histogram')
                result = MNIST_CNN.CNN(train, test, times=train_num_times, feature_name='histogram')
            elif choose_feature == "轮廓特征":
                train = mnist
                test = feature_extract(test_data, test_num, 'contour')
                result = MNIST_CNN.CNN(train, test, times=train_num_times, feature_name='contour')
            elif choose_feature == "偏移校正":
                train = mnist
                test = feature_extract(test_data, test_num, 'correct')
                result = MNIST_CNN.CNN(train, test, times=train_num_times, feature_name='correct')
            elif choose_feature == "图像锐化":
                train = mnist
                test = feature_extract(test_data, test_num, 'sharpening')
                result = MNIST_CNN.CNN(train, test, times=train_num_times, feature_name='sharpening')
            elif choose_feature == "原始图像":
                train = mnist
                test = test_data
                result = MNIST_CNN.CNN(train, test, times=train_num_times, feature_name='original')
            else:
                result = (False, None)
                print('您选择的特征不适合在此模型上运行，请重新选择！')

            if result[0] == True:
                print('CNN(卷积神经网络)分类完毕！')
                print('若想要二次构建模型，请退出后重新运行程序，谢谢啦！')
            else:
                print('CNN(卷积神经网络)运行错误，请检查参数设置！')



if __name__ == "__main__":
    import dataload.load_MNIST_data as MNIST
    from feature import histogram, contour, correct, sharpening, subsampling
    from model import MNIST_KNN, MNIST_Softmax, MNIST_CNN

    mnist = MNIST.load_MNIST()
    train_data_show = mnist.train.next_batch(4)

    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())