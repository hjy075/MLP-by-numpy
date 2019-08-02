from cost import *
import numpy as np

class MulLayer:
    def __init__(self):
        self.x1 = None
        self.x2 = None

    def forward(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        out = x1*x2
        return out

    def backward(self, dout):
        dx1 = dout*self.x2
        dx2 = dout*self.x1
        return dx1, dx2

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x1, x2):
        out = x1 + x2
        return out

    def backward(self, dout):
        dx1 = dout*1
        dx2 = dout*1
        return dx1, dx2

class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1+np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout*(1.0 - self.out)*self.out
        return dx


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

class softmaxwithcee:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = CEE(self.y, self.t)
        return self.loss

    def backword(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t)/batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx/batch_size
        return dx