运行方式: make && ./main
实现功能：用tma从一个大小为[M,2N]的行优先矩阵中，复制前N列到目标矩阵，写成pytorch形式就是: out = in[:,N]


Execution:
make && ./main

Functionality:
This program uses TMA (Tensor Memory Access) to copy the first N columns from a source matrix of size [M, 2N] (in row-major layout) to a destination matrix.

In PyTorch notation, this operation is equivalent to:
out = in[:, :N]