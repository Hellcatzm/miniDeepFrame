import numpy as np
import Layer
import Loss
import Activate
from tensorflow.examples.tutorials.mnist import input_data

BARCH_SIZE = 4
LEARNING_RATE = 0.01

mnist = input_data.read_data_sets('../../Mnist_data/', one_hot=True)
X_train,y_train = mnist.train.next_batch(BARCH_SIZE)
X_train = np.reshape(X_train, [4, 1, 28, 28])

conv1 = Layer.Conv2D([8, 1, 2, 2])
pool1 = Layer.MeanPooling([2, 2])
relu1 = Activate.Relu()

conv2 = Layer.Conv2D([16, 8, 2, 2])
pool2 = Layer.MeanPooling([2, 2])
relu2 = Activate.Relu()

conv3 = Layer.Conv2D([8, 16, 2, 2])
pool3 = Layer.MeanPooling([2, 2])
relu3= Activate.Relu()

dense1 = Layer.Dense(128, 10)
sigmoid = Activate.Sigmoid()

loss = Loss.MSECostLayer()

loss_line = []
for i in range(1000):
    # 正向传播
    x = conv1.forward(X_train, 1)
    x = pool1.forward(x, 2)
    x = relu1.forward(x)
    x = conv2.forward(x, 1)
    x = pool2.forward(x, 2)
    x = relu2.forward(x)
    x = conv3.forward(x, 1)
    x = pool3.forward(x, 2)
    x = relu3.forward(x)
    shape = x.shape
    x = x.reshape([x.shape[0], -1])
    x = dense1.forward(x)

    l_val = loss.loss(x, y_train)
    print("损失函数值为：", l_val)
    loss_line.append(l_val)

    # 反向传播
    l = loss.loss_grad(x, y_train)
    l = dense1.backward(l)
    l = l.reshape(shape)
    l = relu3.backward(l)
    l = pool3.backward(l)
    l = conv3.backward(l)
    l = relu2.backward(l)
    l = pool2.backward(l)
    l = conv2.backward(l)
    l = relu1.backward(l)
    l = pool1.backward(l)
    l = conv1.backward(l)

    for layer in [conv1, conv2, conv3, dense1]:
        for param, param_grad in zip(layer.params(), layer.params_grad()):
            param -= LEARNING_RATE * param_grad

import matplotlib.pyplot as plt

plt.plot(list(range(len(loss_line))), loss_line)
plt.show()
