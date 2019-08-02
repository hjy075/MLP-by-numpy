# MLP-by-numpy
밑바닥부터 딥러닝 구현 by "Jo Hui Senung"

numpy: modeling\
tensorflow: use to load example dataset\
matplotlib: visualizing\
\
layers.py\
  |--add, mul, affine\
  |--relu, sigmoid,softmax-cross entropy\
  \
optimizer.py\
  |--SGD: standard gradient descent\
  |--adam: Adaptive Moment estimation\ 
  \
cost.py\
  |--MSE: mean square error\
  |--cross entropy\
\
model.py\
|--multi layer full connected(+regularization: L1, L2)
\
preprocessing.py\
  |--img: faltten, RGB2Gray
