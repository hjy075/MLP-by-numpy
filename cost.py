import numpy as np

def MSE(y_predict, y_true):
    return 0.5*np.sum((y_true - y_predict)**2)

def CEE(y_predict, y_true): #cross_entropy_error
    if y_predict.ndim == 1:
        y_true = y_true.reshape(1, y_true.size)
        y_predict = y_predict.reshape(1, y_predict.size)

    if y_true.size == y_predict.size:
        y_true = y_true.argmax(axis=1)

    batch_size = y_predict.shape[0]
    delta = 1e-7 #log(0)처리
    return -np.sum(np.log(y_predict[np.arange(batch_size), y_true]+delta))/batch_size