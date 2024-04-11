# Deep Learning

## ch05 誤差逆伝播法

NN = ニューラルネットワーク(Neural Network)

- backward 処理

  - step 1: forward で求めた $\bold{y}, L$ を利用する

    - Relu 関数: $h(x) = 0 (x \le 0), x (0 \lt x)$
    - $y = [y_1, y_2]$: Vector
    - $L = \frac{1}{2} ((t_1 - y_1)^2 + (t_2 - y_2)^2)$: Scalar
    - base $y = h(x)(wx + b)$, $y_2 = w_2y_1 + b_2$
    - $L = \frac{1}{2} ((t_1 - h(x_1)(w_1x_1 + b_1))^2 + (t_2 - h(y1)(w_2y_1 + b_2))^2)$: Scalar
    - $L = \frac{1}{2} ((t_1 - (w_1x_1 + b_1))^2 + h(w_1x_1 + b_1)(t_2 - (w_2h(w_1x_1 + b_1) + b_2))^2)$: Scalar → 2 層 NN の損失関数 ($y_1$に$wx + b$を代入)

  - step 2: $\nabla L$を求めて、重み $w$ を更新する

    - $\nabla L = [\frac{∂ L}{∂ w_1}, \frac{∂ L}{∂ w_2}]^T$: Vector (T = Transpose)
    - $L = \frac{1}{2} ((t_1 - h(x_1)(w_1x_1 + b_1))^2 + h(w_1x_1 + b_1)(t_2 - (w_2(w_1x_1 + b_1) + b_2))^2)$ より、一項目の微分は、
    - $(t_1 - h(x_1)(w_1x_1 + b_1)) = x$ とおくと $1/2 (x^2) = x * dx/dw_1$

    - 公式: $d(ax + b)/ dx = (a(x + h)+ b - (ax + b))/h = ah/h = a$ より

    - $d(x_1w_1 + b_1)/ dw_1 = (x_1(w_1 + h)+ b_1 - (x_1w_1 + b_1))/h = x_1h/h = x_1$
    - $\frac{∂ L}{∂ w_1}$ (1 項目)$(t_1 - h(x_1)(w_1x_1 + b_1)) * (h(x_1) * (x_1+ 0)) = (t_1 - h(x_1)(w_1x_1 + b_1)) * h(x_1) x_1$
