# Deep Learning

## ch06 畳みニューラルネットワーク

- Convocational Neural Network (CNN)
  - 畳み込み演算を利用したニューラルネットワーク
  - 特徴: 画像データ内のピクセルがずれると、二次元配列を一次元に変換する時に影響を及ぼす。
    畳み込み演算を行うことで、ピクセルの位置がずれても認識精度を高くできる。
    　- [畳み込み演算の方法](https://data-analytics.fun/2021/11/23/understanding-convolution/)
    　- キーワード: 畳み込み演算、ストライド、パディング、プーリング

## ch07 勾配の最適化(研究)

- この章では、勾配$\nabla L$の最適化計算に関する研究紹介

  - [まとめサイト](https://qiita.com/omiita/items/1735c1d048fe5f611f80)
    ![Optimizer Animation](https://ruder.io/content/images/2016/09/saddle_point_evaluation_optimizers.gif)
    > Image Credit: [Alec Radford](https://twitter.com/alecrad)
  - 実装は ch07 の common/optimizer.py 参照。
  - **最急降下法**: 全データ を使って損失関数が最小値になるように、勾配を使ってパラメータ更新するよ。 = 数値微分
  - **SGD**: データ 1 つだけをサンプルして使うことで、最急降下法にランダム性を入れたよ。
  - **ミニバッチ SGD**: データを 16 個とか 32 個とか使うことで並列計算できるようにした SGD だよ。
  - **モーメンタム**: SGD に移動平均を適用して、振動を抑制したよ。= 勾配を「移動平均」している

    $\nu*t = \beta\nu*{t-1} + (1-\beta)\nabla*w\mathcal{L}(w) \\
    w_t = w*{t-1} - \alpha \nu_t$

    $\nu_{t-1}, \nabla_w\mathcal{L}(w), \alpha$ はそれぞれ前回の勾配、今の勾配、学習率です。$\beta$ は 0 から 1 の値を取るハイパーパラメータです。

  - **NAG**: モーメンタムで損失が落ちるように保証してあるよ。
  - **RMSProp**: 勾配の大きさに応じて学習率を調整するようにして、振動を抑制したよ。

    $\nu_t = \beta\nu_{t-1} + (1-\beta)G^2 \\
    w_t = w_{t-1} - \frac{\alpha}{\sqrt{\nu_t + \epsilon}} G$

  - **Adam**: モーメンタム + RMSProp だよ。今では至る所で使われているよ。

    $\nu_{t} = \beta_1\nu_{t-1} + (1-\beta_1)G \\
    s_{t} = \beta_2s_{t-1} + (1-\beta_2)G^2 \\
    w_t = w_{t-1} - \alpha\frac{\nu_{t}}{\sqrt{s_t + \epsilon}}$

  - **ニュートン法**: 二階微分 を使っているので、ものすごい速さで収束するよ。ただ計算量が膨大すぎて実用されていないよ。
    $$w_{t+1} = w_{t} - \frac{\mathcal{L'}(w)}{\mathcal{L''}(w)}$$

    一般には、
    $$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

    → 有限要素法で使用されている
    $f(x) = 0$を解くアルゴリズム

    - 例 1: $f(x) = x^3 - 1 = 0$　の解

      - $f'(x) = 3x^2$より、

      - $x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} = x_n - \frac{x_n^3 - 1}{3x_n^2}$

    - 例 2: $f(x) = x^2 - 1$ の解

      - $f'(x) = 2x$より、

      - $x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} = x_n - \frac{x_n^2 - 1}{2x_n}$

      - $\sqrt{2}$ の値を求めるプログラム

- [Playground(Python3)](https://godbolt.org/z/P69eMcfvK)
- [Playground(Rust)](https://godbolt.org/z/4sqKcs6eq)

```python
# 適当な初期値の設定
x = 5.0
while True:
    # ニュートン法による新しいxを求める
    x2 =  x - (x * x - 2) / (x * 2)

    # 計算後の値が誤差の範囲内になったら計算終了
    if abs(x2 - x) < 0.0001:
        break

    # 計算後の値をxとして計算を繰り返す
    x = x2

    # 計算結果の表示
    print(x)
```

### 移動平均とは？

移動平均とは経済でよく使われ、急な変化があるグラフに対して移動平均を用いるとその変化がゆるやかになったグラフが得られる優れものです。(正確には指数平滑移動平均という仰々しい名前がついていますが、ここではわかりやすく移動平均に統一します。)

![移動平均のグラフ](https://imgur.com/BR6Axhp.png)

オリジナルの青線に対して移動平均を用いているのが赤線のグラフ。
振動を抑え、緩やかになっていることがわかります。

つまり、**移動平均は急な変化に動じないグラフになっている** ことがわかります。
この赤線は以下の式で各点を求め、プロットしています。

$$
\nu_t = \beta\nu_{t-1} + (1-\beta)G
$$

例: $$\nu_t = 0.9 * 5 + 0.1 * G$$

ここで $\nu_{t-1}, G$ はそれぞれ前時刻での移動平均された後の値、G は現時刻の値で、$\beta$ は 0 から 1 の値を取るハイパーパラメータです。右辺において、

- **第 1 項 $\beta\nu_{t-1}$ : 今までの $G$ たちを蓄積した項**
- **第 2 項 $(1-\beta)G$ : 現在の点を表す項**

ということがわかります。$\beta$ が大きければ大きいほど第 1 項つまり今までの値の影響が大きく、第 2 項つまり現時刻の影響が小さくなります。
