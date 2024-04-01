# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    # dy/dx = df/dx
    # (f(x + h) - f(x) + f(x) + f(x - h)) / (2 * h)
    dydx = (f(x+h) - f(x-h)) / (2*h)  # numerical
    # dydx = 0.02 * x + 0.1  # theory
    return dydx


def function_1(x):
    return 0.01*x**2 + 0.1*x 


def tangent_line(f, x):
    # differential
    d = numerical_diff(f, x)
    print(d)
    # y = f(x + dx) = f(x) + {(f(x+h) - f(x-h)) / (2*h)} * x + ...
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
