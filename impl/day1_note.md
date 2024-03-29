# deep learning

## ch01 パーセプトロン

線形代数

- $y = w x + b$
- $y$ : 出力　$w$: 重み　$x$: 入力　$b$: バイアス
- 目的: 0, 1 のコンピュータを作成したい

- 多次元データの場合
- $\bold{y} = \bold{w} \bold{x}+ \bold{b}$ で表せる

## ch02 論理回路

ゲートの種類

- AND
- OR
- NAND
- XOR

- 自動でパラメータを調整したい
- $\bold{y} = h(\bold{x})(\bold{w} \bold{x} + \bold{b})$
- $h(\bold{x})$を活性化関数と呼ぶ

## ch03 画像処理と活性化関数

- $\bold{y} = h(\bold{x})(\bold{w} \bold{x} + \bold{b})$
- $h(\bold{x})$を活性化関数と呼ぶ

活性化関数 $h(\bold{x})$ の代表例

- Relu(Rectified Linear Unit) 関数 (ランプ関数)
- sigmoid 関数 $h(x) = \frac{1}{1 + e^{-x}}$
- step 関数 (階段関数) $h(x) = 0 (x < 0>)$, $h(x) = 1 (x >= 0)$
- ソフトマックス関数 $h(x) = \frac{e^{x_n}}{\Sigma_{k = 1}^{N} {e^{x_k}}}$
