import pickle
from preprocessing import *
import matplotlib.pyplot as plt

with open('network_mnist', 'rb') as f:
    network = pickle.load(f)

(x_train, y_train), (x_test, y_test) = data_load()

pred = np.argmax(network.predict(x_test[:100]), axis=1)

print(x_train[0].shape, pred[0])
