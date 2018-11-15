import numpy as np

class Sigmoid:
    def forward(self, x):
        self.x = x
        return 1./(1.+np.exp(-x))

    def backward(self, x_grad):
        s = self.forward(self.x)
        return s*(1-s) * x_grad

class Tanh:
    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, x_grad):
        e = self.forward(self.x)
        return (1-e**2) * x_grad

class Relu:
    def forward(self, x):
        self.x = x
        return np.maximum(0., x)

    def backward(self, x_grad):
        return np.where(self.x>=0, x_grad, 0)

class Softmax:
    def forward(self, x):
        self.x = x
        return np.maximum(0., x)

    def backward(self, x_grad):
        return np.where(self.x>=0, x_grad, 0)