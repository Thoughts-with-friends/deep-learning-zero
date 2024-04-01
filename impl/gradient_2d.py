# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    # x.size = 2: (x, y)
    new_x = x # [x, y]
    for idx in range(x.size):
        new_x[idx] = float(new_x[idx]) + h
        fxh1 = f(new_x)  # f(x+h)

        new_x[idx] = new_x[idx] - h
        fxh2 = f(new_x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
    
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


def function_2(x):
    # print(np.sum(x**2))
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    
    # [[0, 1], [0, 1]] => [0, 1, 0, 1]
    X = X.flatten()
    Y = Y.flatten()
    
    # z = x^2 + y^2
    # grad(z) = (2x, 2y) = 2 (x, y)
    grad = numerical_gradient(function_2, np.array([X, Y]).T).T
    plt.figure()
    
    # plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="red")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()
