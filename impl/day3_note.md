# Deep Learning

## ch04 深層学習の基本アルゴリズム

<復習>
深層学習(NN)のアルゴリズム

- $forward$

  1.画像データ（教師データ $t$、入力データ $x$）を読み取る

  2.重みパラメータ $W$ を適当に決める。

  3.$y = h(x) × (Wx + b)$ から、出力 $y$ を求める。($h(x)$ は活性化関数)

  4.$k$ 番目のデータ $y_k$, $t_k$ から、誤差関数(損失関数) $L$ を求める。

  $L$ の例: ($N$ はデータ数)

  - $y = [y_1, y_2, y_3]$: Vector
  - $L = \frac{1}{2} ((t_1 - y_1)^2 + (t_2 - y_2)^2 + (t_3 - y_3)^2$: Scalar
  - 平均二乗誤差 $L = \frac{1}{2} \Sigma^{N}_{k = 1} (t_k - y_k)^2$
  - $n$ 層ニューラルネットワークの平均二乗誤差: $L = \frac{1}{2} \Sigma^{N}_{k = 1} (t_k - y_{k, n}(y_{k, n-1}(y_{k, n-2}(...(y_{k,1})))))^2$
  <!-- $y[k][n] = y[row][column]$ -->
  - 交差エントロピー誤差 $L = - \Sigma^{N}_{k = 1} t_k log_{10}(y_k)$
  - ポイント: $t_k - x_k$だ常に$L$が一定になってしまうから。

    - 誤差関数が$L$や$W$と関係ない(e.g. $x^2$など)だったら$\frac{dL}{dW}$で微分しても関係がないから$0$になってしまう
    - $\nabla L = \frac{\partial L}{\partial  W} = 0$
    - 求めたいのは W について微分であって x についての微分ではないから。

- $backward$ ($forward$で求めた$L$を使って、層の「逆順」に重みを更新する)

  5.微分演算子 $\nabla$ を用いて、$L = L(W1, W_1, W_2, ..., W_n)$ の微分 $\nabla L = \frac{\partial L}{\partial  W}$ を計算する(疑似コード参照)。

  6.学習率 $lr$ を調整しながら、重みパラメータ$W$ を $W_{new} = W_{old} - lr \nabla L$ で更新する。

  7.$L$ が最小値を取るまで、3 ～ 6 を繰り返す。

  8.最小値 $L$ を取る時の $W$ が、学習後の重みパラメータ $W$ となる。

```rust
// y = y(x)について
// 意味: yはxの関数であることを明記するため
let y;
let _y = x^2 + x + 1;

// y = y(x) = any_func(x)
y = _y;
```

- 誤差関数を計算例

```rust
fn mean_squared_error(t: f32, y: f32) -> f32 {
    0.5 * (t - y) * (t - y)
}

fn cross_entropy_error(t: f32, y: f32) -> f32 {
    - t * log(y)
}

/// 誤差関数でk番目の誤差(L)を求める。
// L_K = L_k(W_k)
fn get_L_k(W: Matrix3) -> [f32; 3] {
    x = [1, 2, 3];
    b = [1, 1, 1];

    // 内積を計算する
    // b が3要素あるからyも3要素になる
    // y = [a, b, c]
    y = dot_product(W, x) + b;

    // k番目の平均二乗誤差と
    // 交差エントロピー誤差
    // out_L_k = mean_squared_error(t, y);
    out_L_k = cross_entropy_error(t, y);

    out_L_k
}

// k番目の誤差を標準出力する疑似コード
fn main()
{
    // データ数
    let n = 100;

    // k番目の重み
    let W_k = [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ] * n;

    let mut L = 0.0;

    // 交差エントロピー誤差
    // $L = - \Sigma^{N}_{k = 1} t_k log_{10}(y_k)$
    // l = for (k = 1..n) L_k;
    for k in 1..n
    {
        L += get_L_k(W_k);
    }

    println!(L);
}
```

- 例） 平均二乗誤差 $L$ の場合

  - $L \\
     = \frac{1}{2} \Sigma^{N}_{k = 1} (t_k - y_k)^2 \\ = \frac{1}{2} \Sigma^{N}_{k = 1} (t_k - (W_k x_k + b_k))^2$

## ch05 誤差逆伝播法

- 誤差逆伝播法 = 微分計算 $W_{new} = W_{old} - lr \nabla L$ を行う最適化アルゴリズムの一つ。
- 誤差逆伝搬法、back propagation ともいう。
- 微分計算を解析的に（厳密に）求めておき、再利用することで速く・正確に計算できる。
- Relu 関数: $h(x) = x (0 \le x)$ の場合:
  $\frac{dh}{dx} = 1 (0 \le x), \frac{dh}{dx} = 0 (x < 0)$

- 合成関数の微分
- 合成関数 = y = f(g(x))
  $\frac{dy}{dx} = \frac{df}{dg}\frac{dg}{dx}$
- 証明:
  $h_g = g(x + h) - g(x), g = g(x)$ とおく。
  $lim_{h \rightarrow 0} \frac{df}{dx}
  \\ = lim_{h \rightarrow 0}  \frac{f(g+ h_g) - f(g)}{h}
  \\ = lim_{h \rightarrow 0}  \frac{f(g+ h_g) - f(g)}{h_g} * \frac{h_g}{h}
  \\ = lim_{h \rightarrow 0}  \frac{f(g(x)+ h_g) - f(g(x))}{h_g} * \frac{g(x + h) - g(x)}{h}
  \\ = lim_{h \rightarrow 0}  \frac{f(g(x)+ h_g) - f(g(x))}{g(x + h) - g(x)} * \frac{g(x + h) - g(x)}{h}$

- 【合成関数の微分】
  <!-- \large \huge -->

  $\large{\frac{dy}{dx} = \frac{df}{dg}\frac{dg}{dx}}$
  $= \frac{df}{dh} \frac{dh}{dg} \frac{dg}{dx}$

  一般に、
  $\frac{dy}{dx} = \frac{df_1}{df_2} \frac{df_2}{df_3} ... \frac{df_{n-1}}{df_{n}} \frac{df_{n}}{dx}$

- $f(x) = e^x: \frac{df}{dx} = \frac{d(e^x)}{dx} = e^x$
- $f(g) = e^g: \frac{df}{dg} = \frac{d(e^g)}{dg} = e^g$
- (問 1) $y = e^{x^2}$の微分を求めよ。
  (解答) $x^2 = g$とおくと、$y = e^g$とおける。
  $\frac{dy}{dx} = \frac{dy}{dg}\frac{dg}{dx} = \frac{dg}{dx} \frac{d(e^g)}{dg} =  \frac{dg}{dx} e^g$
  よって、
  $\frac{dy}{dx} = (2x) * e^{x^2}$

- (問 2) $f(x) = sinxのとき、df/dx = cosx$を用いて、以下の微分を求めよ:
  $y = sin({x^2})$
- (解答) $2x * cos{(x^2)}$
