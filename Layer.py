import numpy as np


class Layer:
    def forward(self, *arge):
        raise NotImplementedError()

    def backward(self, *arge):
        raise NotImplementedError()

    def params(self):
        raise NotImplementedError()

    def params_grad(self):
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, in_units, out_units, weight=None, bias=None):
        self.in_units = in_units
        self.out_units = out_units
        # 初始化参数
        if weight is not None:
            assert weight.shape == (in_units, out_units), "权重初始值有误"
            self.w = weight
        else:
            self.w = np.random.normal(size=(in_units, self.out_units))
        if bias is not None:
            assert bias.shape == (out_units,), "偏移初始值有误"
            self.b = bias
        else:
            self.b = np.zeros(self.out_units)

    def forward(self, input_tensor):
        assert np.ndim(input_tensor) == 2 and \
               input_tensor.shape[1] == self.in_units, 'Input Tensor should with shape [batch, in_units]'
        self.last_input = input_tensor
        return np.dot(self.last_input, self.w) + self.b

    def backward(self, output_grad):
        self.grad_w = np.dot(self.last_input.T, output_grad)/output_grad.shape[0]
        self.grad_b = np.mean(output_grad, axis=0)
        return np.dot(output_grad, self.w.T)

    def params(self):
        return self.w, self.b

    def params_grad(self):
        try:
            return self.grad_w, self.grad_b
        except BaseException as e:
            print("还没有生成梯度信息！")


class Conv2D(Layer):
    def __init__(self, kernal_shape):
        """
        kernal_shape: [out_depth, in_depth, h, w]
        """
        self.kernal = np.random.normal(size=kernal_shape)
        self.bias = np.zeros(shape=[kernal_shape[0], ])

        self.kernal_grad = np.zeros(shape=kernal_shape)
        self.bias_grad = np.ones(shape=[kernal_shape[0], ])

    def forward(self, input_tensor, stride, padding="SAME"):
        """
        input_tensor: [batch, channel, h, w]
        stride: scalar
        padding: {"SAME", "VALID"}
        """
        self.stride = stride
        batch, _, i_h, i_w = input_tensor.shape
        out_d, _, k_h, k_w = self.kernal.shape
        self.i_h, self.i_w = i_h, i_w
        if padding == "SAME":
            out_h = np.ceil(i_h / stride).astype(np.int)
            out_w = np.ceil(i_w / stride).astype(np.int)
            pad_h = (k_h - 1) / 2.
            pad_w = (k_w - 1) / 2.
            input_tensor = np.pad(input_tensor,
                                  ((0, 0),
                                   (0, 0),
                                   (int(np.floor(pad_h)),
                                    int(np.ceil(pad_h))),
                                   (int(np.floor(pad_w)),
                                    int(np.ceil(pad_w)))),
                                  mode='constant')
        elif padding == "VALID":
            out_h = np.ceil((i_h - k_h + 1) / 2).astype(int)
            out_w = np.ceil((i_w - k_w + 1) / 2).astype(int)
        else:
            print("padding must be one of {} and {}".format("SAME", "VALID"))

        self.input_tensor = input_tensor
        self.output_tensor = np.empty(shape=[batch, out_d, out_h, out_w])

        for b in range(batch):
            for d in range(out_d):
                for h in range(out_h):
                    for w in range(out_w):
                        self.output_tensor[b, d, h, w] = np.sum(
                            self.kernal[d] * input_tensor[b, :,
                                             h * stride:h * self.stride + k_h,
                                             w * stride:w * self.stride + k_w],
                            keepdims=False)
            self.output_tensor[b, d] += self.bias[d]
        return self.output_tensor

    def backward(self, output_grad):
        """
        output_grad: [batch, out_d, out_h, out_w]
        """
        batch, _, i_h, i_w = self.input_tensor.shape
        _, out_d, out_h, out_w = output_grad.shape
        _, in_d, k_h, k_w = self.kernal.shape
        input_grad = np.zeros(shape=self.input_tensor.shape)
        for b in range(batch):
            for d in range(out_d):
                for h in range(out_h):
                    for w in range(out_w):
                        input_grad[b, :,
                        h * self.stride: h * self.stride + k_h,
                        w * self.stride: w * self.stride + k_w] += output_grad[b, d, h, w] * self.kernal[d]
                        self.kernal_grad[d, :, :, :] += output_grad[b, d, h, w] * self.input_tensor[b, :,
                                                                                  h * self.stride: h * self.stride + k_h,
                                                                                  w * self.stride: w * self.stride + k_w]
            self.bias_grad[d] += np.mean(output_grad[b, d])
        return input_grad[:, :, :self.i_h, :self.i_w]

    def params(self):
        return self.kernal, self.bias

    def params_grad(self):
        return self.kernal_grad, self.bias_grad


class MeanPooling(Layer):
    def __init__(self, kernal_shape):
        """kernal_shape: [h, w]"""
        self.kernal_shape = kernal_shape

    def forward(self, input_tensor, stride, padding="SAME"):
        """
        input_tensor: [batch, channel, h, w]
        stride: scalar
        padding: {"SAME", "VALID"}
        """
        self.stride = stride
        batch, i_d, i_h, i_w = input_tensor.shape
        k_h, k_w = self.kernal_shape
        self.i_h, self.i_w = i_h, i_w
        if padding == "SAME":
            out_h = np.ceil(i_h / stride).astype(np.int)
            out_w = np.ceil(i_w / stride).astype(np.int)
            pad_h = (k_h - 1) / 2.
            pad_w = (k_w - 1) / 2.

            # mask标记图片和pad部分
            kernal = np.ones(shape=[i_h, i_w])
            kernal = np.pad(kernal,
                            ((int(np.floor(pad_h)),
                              int(np.ceil(pad_h))),
                             (int(np.floor(pad_w)),
                              int(np.ceil(pad_w)))),
                            mode='constant')

            input_tensor = np.pad(input_tensor,
                                  ((0, 0),
                                   (0, 0),
                                   (int(np.floor(pad_h)),
                                    int(np.ceil(pad_h))),
                                   (int(np.floor(pad_w)),
                                    int(np.ceil(pad_w)))),
                                  mode='constant')

        elif padding == "VALID":
            # mask标记图片和pad部分
            kernal = np.ones(shape=[i_h, i_w])

            out_h = np.ceil((i_h - k_h + 1) / 2).astype(int)
            out_w = np.ceil((i_w - k_w + 1) / 2).astype(int)
        else:
            print("padding must be one of {} and {}".format("SAME", "VALID"))

        self.kernal = kernal
        self.input_tensor = input_tensor
        self.output_tensor = np.empty(shape=[batch, i_d, out_h, out_w])

        for b in range(batch):
            for d in range(i_d):
                for h in range(out_h):
                    for w in range(out_w):
                        # 利用mask信息计算每一个位置的kernal值
                        cur_kernal = kernal[h * self.stride:h * self.stride + k_h,
                                     w * self.stride:w * self.stride + k_w]
                        cur_kernal = cur_kernal / np.sum(cur_kernal)

                        self.output_tensor[b, d, h, w] = np.sum(
                            cur_kernal *
                            input_tensor[b, d,  # d的位置最初误写作:，这和conv不同
                            h * self.stride:h * self.stride + k_h,
                            w * self.stride:w * self.stride + k_w],
                            keepdims=False)
        return self.output_tensor

    def backward(self, output_grad):
        """
        output_grad: [batch, out_d, out_h, out_w]
        """
        input_grad = np.zeros_like(self.input_tensor, dtype=np.float32)
        batch, out_d, out_h, out_w = output_grad.shape
        k_h, k_w = self.kernal_shape
        for b in range(batch):
            for d in range(out_d):
                for h in range(out_h):
                    for w in range(out_w):
                        # 利用mask信息计算每一个位置的kernal值
                        cur_kernal = self.kernal[h * self.stride:h * self.stride + k_h,
                                     w * self.stride:w * self.stride + k_w]
                        cur_kernal = cur_kernal / np.sum(cur_kernal)

                        input_grad[b, :,
                        h * self.stride:h * self.stride + k_h,
                        w * self.stride:w * self.stride + k_w] \
                            += output_grad[b, d, h, w] * np.broadcast_to(cur_kernal,
                                                                         [out_d, *cur_kernal.shape])
        return input_grad[:, :,
               int(np.floor((k_h - 1) / 2)):self.i_h + int(np.floor((k_h - 1) / 2)),
               int(np.floor((k_w - 1) / 2)):self.i_w + int(np.floor((k_w - 1) / 2))]

    def params(self):
        pass

    def params_grad(self):
        pass


if __name__ == "__main__":
    dense = Dense(3, 5, bias=np.ones(5))
    x = np.ones([1, 3])
    dense.forward(x)
