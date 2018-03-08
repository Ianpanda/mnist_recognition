# coding:utf-8 #
__author__ = 'Bai Bo'

def load_MNIST():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    return mnist

if __name__ == "__main__":
    mnist_data = load_MNIST()
    if mnist_data != None:
        print('MNIST手写数据集读取成功！')
    else:
        print('读取失败！')