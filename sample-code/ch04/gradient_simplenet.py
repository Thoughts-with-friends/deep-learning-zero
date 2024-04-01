# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        # 適当なW[2][3]を決める
        self.W = np.random.randn(2,3)

    # z[3] = X[2] * W[2][3]
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        # print(f"z = {z}")
        
        # 活性化関数に代入
        # Note: この関数が代表例なだけでこれを使わないといけないわけではない
        y = softmax(z)
        # print(f"y = {y}")
        
        # クロスエントロピー誤差
        # L = -sum(t_k log_10 (y_k))
        # loss = scalar: float64
        loss = cross_entropy_error(y, t)
        # print(loss)

        return loss

# 1. 入力x、教師データtを決める
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

# 2. Wを適当に決定
net = simpleNet()

# 3. y = wx + bを計算し、交差エントロピー誤差fを求める
f = lambda _w: net.loss(x, t)
# print(f"net.w = {net.W}")

# 4. Wを+hしながら、誤差f = f(W)を
#   微分df/dWで評価して、(Wを)更新する
dW = numerical_gradient(f, net.W)

print(dW) # 最適なweight
