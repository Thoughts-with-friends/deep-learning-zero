# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

X = np.arange(-100.0, 100.0, 1)
Y = softmax(X)
plt.plot(X, Y)
# plt.ylim(-0.1, 1.1)  # 図で描画するy軸の範囲を指定
plt.show()