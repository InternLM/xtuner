
TP原理是矩阵乘的权重按列或者按行放在不同的卡上。
具体的计算流程： 
以TP=2为例，输入X [N1, D_h] 先经过AllGather得到 [N1+N2, D_h]，
再经过ColumnParallelLinear（即原始权重 A 按列分 [A1, A2]，每个 rank 一份权重，比如 rank0 持有 A1 权重）的矩阵乘法得到中间分片的Y，
再经过RowParallelLinear（即原始权重 B 按行分，每个 rank 一份权重，比如 rank0 持有 B1 权重）的矩阵乘法，
然后做reduce scatter得到最终输出Z。

这在数学上和普通MLP是等价的

$$
\begin{bmatrix}
X1 \\
X2
\end{bmatrix}  \times 
\begin{bmatrix}
A1 & A2
\end{bmatrix}
= 
\begin{bmatrix}
Y1 & Y2
\end{bmatrix}
$$

$$
\begin{bmatrix}
Y1 & Y2
\end{bmatrix} \times \begin{bmatrix}
B1 \\
B2
\end{bmatrix} = Y1B1+Y2B2 
$$
