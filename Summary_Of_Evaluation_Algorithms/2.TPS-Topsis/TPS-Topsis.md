# Topsis法

## 矩阵正向化

### 指标类型

1.极大型(利润型) 越大越好

处理方法:直接找最大值即可

2.极小型(成本型) 越小越好

处理方法:

\[
\hat{x} = max - x \qquad \begin{cases}
\hat{x}:处理后数据 \\
max:最大值 \\
x:处理前数据
\end{cases}
\]

3.中间型(Ph值) 越靠近某个值越好

处理方法:

\[
\hat{x} = 1 - \frac{|x_i - x_{best}|}{M} \qquad M = max[|x_i - x_{best}|] \qquad \begin{cases}
\hat{x}:处理后数据 \\
x_{best}:最优值 \\
x_i:指标值(处理前数据)
\end{cases}
\]

数据越好$\hat{x} \rightarrow 1 $,数据越差$\hat{x} \rightarrow 0 $

4.区间型(温度) 在某个范围内越好

处理方法:

\[
\hat{x} \begin{cases}
1 - \dfrac{a - x_i}{M} \quad  &, \quad x_i < a \\
1 &, \quad a\leq x_i \leq b \\
1 - \dfrac{x_i - b}{M} &, \quad b < x_i
\end{cases}
\qquad M = max[a - min{x_i},max{x_i} - b] \begin{cases}
\hat{x}:处理后数据 \\
[a,b]:最佳区间 \\
x_i:指标值(处理前数据)
\end{cases}
\]

## 矩阵标准化

为了消除不同指标的量纲

\[
z_{ij} = \frac{x_{ij}}{\sqrt{\sum^n_{i=1} x^2_{ij}}} \qquad \begin{cases}
z_ij:处理后数据 \\
x_ij:处理前数据
\end{cases}
\]

## 求值

\[
D_+ = \sqrt{\sum^m_{j=1}w_j(Z^+_j - z_{ij})^2} \\[8pt]
D_- = \sqrt{\sum^m_{j=1}w_j(Z^-_j - z_{ij})^2} \qquad \begin{cases}
\end{cases}
w_j:权重
\\[8pt]
S_i = \frac{D_i^-}{Di^+} + D_i^-
\]
