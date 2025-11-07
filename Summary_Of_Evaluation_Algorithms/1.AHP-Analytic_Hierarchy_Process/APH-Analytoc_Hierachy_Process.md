# 层次分析法

## 权重计算计算方法

### 1.算术平均法:

\[
W_i = \frac{1}{n} \sum_{j = 1}^n \frac{a_{ij}}{\sum_{k = 1} ^ n} \; (i = 1,2,3 \dots\,n)
\]

归一化：按列求和，元素除于求和的结果，再按行求平均值

### 2.几何平均法：

\[
W_i = \frac{(\prod_{j=1}^na_{ij})^\frac{1}{n}}{\sum_{k=1}^n(\prod_{j=1}^na_{kj})^\frac{1}{n}} \;(i=1,2,3,\dots n)
\]

按行相乘开n次方，得到一列数据，再归一化

###  3.特征值法：
把最大特征值的特征向量归一化得权重

## 一致性检验

\[
CI = \frac{\lambda_{max} - n}{n - 1} \qquad RI = \frac{\lambda^\prime_{max} - n}{n-1}
\]

\[
CR = \frac{CI}{CR} \; \begin{cases}
0 \quad & 为一致 \\
< 0.1 & 满足一致 \\
\geq 0.1 & 不满足一致
\end{cases}
\]
