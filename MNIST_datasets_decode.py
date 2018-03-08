#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Bai Bo and Jia Chen'
from PIL import Image
import struct, gzip

def uncompress_file(fn_in, fn_out):  
    f_in = gzip.open(fn_in, 'rb')  
    f_out = open(fn_out, 'wb')  
    file_content = f_in.read()  
    f_out.write(file_content)  
    f_out.close()  
    f_in.close() 

def read_image(filename, type = 1):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()

    magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')

    if type == 1:
        for i in range(images):
            image = Image.new('L', (columns, rows))
            for x in range(rows):
                for y in range(columns):
                    image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                    index += struct.calcsize('>B')

            print('save ' + str(i+1) + 'image')
            image.save(r'./MNIST_data/MNIST/train' + '/'+ str(i+1) + '.png')
    elif type == 2:
        for i in range(images):
            image = Image.new('L', (columns, rows))
            for x in range(rows):
                for y in range(columns):
                    image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                    index += struct.calcsize('>B')

            print('save ' + str(i + 1) + 'image')
            image.save(r'./MNIST_data/MNIST/test' + '/' + str(i + 1) + '.png')


def read_label(filename, saveFilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()

    magic, labels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    labelArr = [0] * labels

    for x in range(labels):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')

    save = open(saveFilename, 'w')

    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')
    save.close()
    print('save labels success')


if __name__ == '__main__':
    gzip_image_train = r'./MNIST_data/train-images-idx3-ubyte.gz'
    gunzip_image_train = r'./MNIST_data/train-images.idx3-ubyte'
    gzip_labels_train = r'./MNIST_data/train-labels-idx1-ubyte.gz'
    gunzip_labels_train = r'./MNIST_data/train-labels.idx1-ubyte'
    uncompress_file(gzip_image_train, gunzip_image_train)
    uncompress_file(gzip_labels_train, gunzip_labels_train)
    gzip_image_test = r'./MNIST_data/t10k-images-idx3-ubyte.gz'
    gunzip_image_test = r'./MNIST_data/t10k-images.idx3-ubyte'
    gzip_labels_test = r'./MNIST_data/t10k-labels-idx1-ubyte.gz'
    gunzip_labels_test = r'./MNIST_data/t10k-labels.idx1-ubyte'
    uncompress_file(gzip_image_test, gunzip_image_test)
    uncompress_file(gzip_labels_test, gunzip_labels_test)
    read_image(r'./MNIST_data/train-images.idx3-ubyte')
    read_label(r'./MNIST_data/train-labels.idx1-ubyte', r'./MNIST_data/MNIST/train/label.txt')
    read_image(r'./MNIST_data/t10k-images.idx3-ubyte', type=2)
    read_label(r'./MNIST_data/t10k-labels.idx1-ubyte', r'./MNIST_data/MNIST/test/label.txt')
