#library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#함수
def unit(x):
    y = x > 0
    return y.astype(int)

#data
x = np.array([[0, 0],[0,1], [1,0], [1,1]]) # (4 * 2)
#x = np.array([[i, j] for i in range(-10,10) for j in range(-10,10)], dtype = float)


#layer
layer = {}
layer['w1'] = np.array([[-0.5, 0.5], [-0.5, 0.5]]) #(2*2)
layer['b1'] = np.array([0.7, -0.2]) #(1*2) // broad casting
layer['w2'] = np.array([0.5, 0.5]).T #(2*1)
layer['b2'] = np.array([-0.7])

#propa
a1 = np.dot(x, layer['w1']) + layer['b1']
z1 = unit(a1)
a2 = np.dot(z1, layer['w2']) + layer['b2']
z2 = unit(a2)
y_predict = z2


#data만들기
df = pd.DataFrame({'x1' : x[:, 0], 'x2' : x[:,1], 'y' : y_predict})

groups = df.groupby('y')

#그래프 그리기
fig, ax = plt.subplots()
for name, group in groups:
    ax.scatter(group.x1, group.x2, marker='o', label=name)
ax.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()