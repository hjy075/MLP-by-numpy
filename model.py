import numpy as np
from collections import OrderedDict

from layers import *

class TwoAffineNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #가중치 초기화
        self.params = {}
        self.params['w1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['relu1'] = ReLU()
        self.layers['affine2'] = Affine(self.params['w2'], self.params['b2'])

        self.lastlayer = softmaxwithcee()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.lastlayer.backword(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['w1'] = self.layers['affine1'].dw
        grads['b1'] = self.layers['affine1'].db
        grads['w2'] = self.layers['affine2'].dw
        grads['b2'] = self.layers['affine2'].db
        return grads


class MultiAffineNet:
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        self.__init_weight(weight_init_std)

        activation_layer = {'relu' : ReLU}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['w' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation' + str(idx)] = activation_layer[activation]()

            #use_batchnorm:

            #use_dropout:

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['w' + str(idx)],
                                                  self.params['b' + str(idx)])
        self.last_layer = softmaxwithcee()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu'): #He_initializer
                scale = np.sqrt(2.0 / all_size_list[idx-1])
            self.params['w' + str(idx)] = scale*np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            w = self.params['w' + str(idx)]
            weight_decay += 0.5*self.weight_decay_lambda*np.sum(w**2) #L2 regulization
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backword(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['w' + str(idx)] = self.layers['Affine' + str(idx)].dw + 2*self.weight_decay_lambda*self.layers['Affine'+str(idx)].w
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
        return grads