from optimizer import *
from model import *
from preprocessing import *

print("1.mnist, 2.fashion_mnist, 3.cifar10")
dataset_name = numtostr(input("번호를 입력하세요 : "))
#data load
(x_train, y_train), (x_test, y_test) = data_load(dataset_name)
print("x_train.shape = {}".format(x_train.shape))
print("x_test.shape = {}".format(x_test.shape))

iters_num = 6000
train_size = x_train.shape[0]
batch_size = 100
print('batch_size = {}'.format(batch_size))

seed = 1
np.random.seed(seed)

network = MultiAffineNet(input_size=x_train.shape[1], hidden_size_list=[100, 100], output_size=y_train.shape[1], weight_decay_lambda=0.1)
optimizer = SGD(lr=0.1)
#optimizer = Adam()

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size, 1) #1epoch : train데이터 전체를 1번 학습

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,  batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    grad = network.gradient(x_batch, y_batch)

    optimizer.update(network.params, grad)
    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        train_loss = network.loss(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        test_loss = network.loss(x_test, y_test)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        print('{} epoch : train 전체를 {}번 학습한 상태'.format(int(i/iter_per_epoch), i/iter_per_epoch))
        print('loss : {}  acc : {}'.format(train_loss,train_acc))
        print()


import matplotlib.pyplot as plt
plt.plot(train_acc_list, 'b', label='train acc')
plt.plot(test_acc_list, 'g', label='test acc')
plt.ylabel('accuracy')
plt.legend(loc='upper left')
plt.show()

if train_acc > 0.8 and test_acc > 0.8:
    import pickle
    save_name = 'network_'+dataset_name
    with open(save_name, 'wb') as f:
        pickle.dump(network, f)

else:
    print('최종 train_acc = {}, test_acc = {} 다시 생각해보세요.'.format(train_acc, test_acc))