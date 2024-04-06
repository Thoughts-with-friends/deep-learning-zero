# coding: utf-8


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        print(f'- Forward -')
        print(f'(x, y) = {self.x, self.y}')

        return out

    def backward(self, dout):
        # tax (dout = 1)
        # self.x =  200 (apple_price)
        # self.y =  1.1 (tax) 
        # dx = 1 * 1.1 = 1.1 (dapple_price) → dout
        # dy = 1 * 200 = 200 (dtax)

        # Apple (dout = 1.1 == dapple_price)
        # self.x =  100 (apple)
        # self.y =  2 (num)
        # dx = 1.1 * 2 = 2.2 (dapple)
        # dy = 1.1 * 100 = 110 (dapple_num)

        print(f'- Backward -')
        print(f'dout = {dout}, (x, y) = {self.x, self.y}')

        # \nabla out = \nabla (xy) = d(xy) / dout dout/d(x, y)
        dx = dout * self.y  # = d(xy) / dout * dout/dx
        dy = dout * self.x  # = d(xy) / dout * dout/dy

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
