import tensorflow as tf
import numpy as np

def rgb2gray(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
def scailing(img):
    img = img.astype(np.float32)
    img /= 255.0
    return img
def flatten(img):
    return img.reshape(-1, img.shape[1]*img.shape[2])

def pre_color(img):
    return scailing(flatten(rgb2gray(img)))

def pre_bw(img):
    return scailing(flatten(img))

def numtostr(num):
    while(True):
        if num == '1':
            return 'mnist'
        elif num == '2':
            return 'fashion_mnist'
        elif num == '3':
            return 'cifar10'
        else:
            print('다시입력하세요.')

def data_load(str):
    if str == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() #mnist
        x_train = pre_bw(x_train)
        x_test = pre_bw(x_test)
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

    elif str == 'fashion_mnsit':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = pre_bw(x_train)
        x_test = pre_bw(x_test)
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

    elif str == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = pre_color(x_train)
        x_test = pre_color(x_test)
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)
