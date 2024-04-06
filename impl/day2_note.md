# Deep Learning

## ch04 数値微分と勾配降下法

数値微分

- 微分をプログラムで計算する方法
- 差分法: 大きく 3 種類ある
- テイラー展開の次数で、精度比較できる
- 前進差分(1 次)、中心差分(2 次)、後進差分(1 次)

勾配降下法

- 中心差分で gradient を計算し、ラグランジュ乗数法で極値（最大・最小の候補点）を求める手法
- 学習率 $lr$ (learning rate)を上手に調整することで、過学習を防止しながら自動計算が可能

＜復習＞ テイラー展開

- テイラー展開 = マクローリン展開の一般化
  [https://manabitimes.jp/math/570]

- 例: $f(x) = x^3$のマクローリン展開

$f(x) = x^3$
$f'(x) = 3x^2$
$f''(x) = 6x$
$f'''(x) = 6$

となる。よって、
$f(x) = 0 + 0 + 0 + \frac{6}{6} x^3 = x^3$

- $\frac{x^k}{k!}$になる理由

$f(x) = x^n$
$\frac{df}{dx} = (n-1) x^(n-1)$
$\frac{d^2f}{d^2x} = (n-1)(n-2) x^(n-2)$
$\frac{d^3f}{d^3x} = (n-1)(n-2)(n-3) x^(n-3)$
...
$\frac{d^{(n-1)}f}{d^{(n-1)}x} = (n-1)! x^(n-n+1)$

$\frac{d^{(n)}f}{d^{(n)}x} = n!$

$3! = 3 × 2 × 1$

- 一次近似（線形近似）

  - $y = ax + b$ の形
    $f(x) \approx f(0) +  f'(0) x$

- 二次近似
  $f(x) \approx f(0) + f'(0) x + \frac{f''(0)}{2} x^2$

- 三次近似
  $f(x) \approx f(0) + f'(0) x + \frac{f''(0)}{2} x^2 + \frac{f'''(0)}{6} x^3$

- 一般形
  $f(x) = \Sigma^{\infty}_{k=0} f^{(k)}(0)\frac{x^k}{k!}$

- 用途: プログラムで実装できるようにするため
- 例: $e^x = \Sigma^{\infty}_{k=0} \frac{x^k}{k!}$

```rust
fn factorial(n: usize) -> f64 {
    let mut res = 1.0;

    if n == 0 {
        res
    } else {
        let mut n = n;
        while n != 1 {
            n -= 1;
            res *= n as f64; // not const here.
        }
        res
    }
}

fn main()
{
  let mut e = 0.0f64;
  let x = 2.0f64;

  for k in 0..5
  {
    let factorial_k = factorial(k) as f64;
    e += x.powf(k as f64) / factorial_k;
  }

  // e^x
  println!("{}", e);
}
```

- [sample](https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=7131e73cd764668b7e2f494a01a0f7b1)

- 内積

  - 2 つのベクトルの相関関係を面積
    $A * A = |A| * |A|$
    $A * B = |A| * |B| cos(\theta)$

## ch04 勾配降下法

勾配降下法とは

- 学習率 $\eta$ を用いて、$x = x - \eta \nabla L$ で計算する手法

  - この方法は、数学的にはラグランジュの未定乗数法になる

- 損失関数$L$: 画像処理など、誤差のモデルとなる関数

  - 目的: $L$の微分を求めて、最適な$W$を算出したい
  - 代表例: 交差エントロピー誤差（関数）$L = Loss$
  - $L = - \Sigma^{N}_{k = 1} t_k log_{10}(y_k)$
  - $t_k$: 訓練データ$t$の$k$番目、$y_k$: 出力値の$k$番目

- ニューラルネットワークの手順

$y(W) = Wx + b$

1. 入力データ $x$ と教師データ $t$、学習率$\lambda$ を決める
2. 誤差$L = L(y, t) = L(W)$を求める
3. $grad(L) = \nabla L = \frac{\partial L}{\partial \bold{W}}$ を計算する
4. $(W, b)_{new} = (W, b)_{old} - \lambda \nabla L$ で重み $W$ とバイアス $b$ を更新する

## ベクトル解析

- $\nabla$ 演算子について

  - $\nabla$ は特別な記号で、3 つある($grad, div, rot$)
  - $\nabla = grad = (\frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \frac{\partial}{\partial z})$
  - 今回は
    $\nabla = \frac{\partial}{\partial \bold{W}} = (\frac{\partial}{\partial W_1}, \frac{\partial}{\partial W_2}, \frac{\partial}{\partial W_3})$
  - $\bold{v}$はベクトル$(x_1, x_2, x_3)$とする
  - 発散、湧き出し、ダイバージェンス (divergence)
  - $div(\bold{v})$
    - $\nabla \cdot \bold{v} = \frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} + \frac{\partial v_z}{\partial z}$
    - イメージ: 空間(単位面積など)から、外にベクトルが何本伸びるか
    - ベクトルだけでなく、行列を取れる
    - 用途: ナビエストークス方程式(流体の運動方程式)、ニュートン運動方程式、ガウスの発散定理
    - 回転 $rot(\bold{v})$ [https://ja.wikipedia.org/wiki/%E5%9B%9E%E8%BB%A2_(%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E8%A7%A3%E6%9E%90)]
    - 行列 $P$ の場合
    - $\nabla \cdot P = (\frac{\partial P_{xx}}{\partial x} + \frac{\partial P_{xy}}{\partial x} + \frac{\partial P_{xz}}{\partial x}, \frac{\partial P_{yx}}{\partial y} + \frac{\partial P_{yy}}{\partial y} + \frac{\partial P_{yz}}{\partial y}, \frac{\partial P_{zx}}{\partial z} + \frac{\partial P_{zy}}{\partial z} + \frac{\partial P_{zz}}{\partial z})$
    - $\Delta = div(grad(\bold{v}))$はラプラス演算子と呼ばれる
    - $\Delta f(x, y, z) = 0$は、ラプラス方程式と呼ばれる
