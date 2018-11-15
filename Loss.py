import numpy as np


class Loss:
    def loss(self, *args):
        raise NotImplementedError()

    def loss_grad(self, *args):
        raise NotImplementedError()


class MSECostLayer(Loss):
    def loss(self, pred, label):
        """
        pred:
        label:
        """
        return np.sum((pred-label)**2)/pred.shape[0]/2.

    def loss_grad(self, pred, label):
        """
        pred:
        label:
        """
        return pred-label