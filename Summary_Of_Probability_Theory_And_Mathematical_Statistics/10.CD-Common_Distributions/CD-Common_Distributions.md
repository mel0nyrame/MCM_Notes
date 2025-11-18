# 常见分布

## 基础概念

### 随机变量:定义在样本空间上的实值函数

随机变量 = $X = X(w)$

离散型随机变量:函数只能取有限个值(可列出来)

连续性随机变量:函数取值充满某个空间(不可列出来)

### 分布列:离散型随机变量通过分布列表示其分布

X取值为$x_1,x_2\dots,x_n$,X为取$x_i$的概率

$p_i = p(x_i)=P(X=x_i)\quad i=1,2\dots,n$

X的分布列记为 $X \sim \{p_i\}$

|$x$|$x_1$|$x_2$|$\dots$|$x_n$|
|:---:|:---:|:---:|:---:|:---|
|$P$|$p(x_1)$|$p(x_2)$|$\dots$|$p(x_n)$|

### 分布函数

表示累积的分布

$F(x) = P (X \leq x)$

分布函数:$F(x) = \sum_{xi < x}p(x_i)$

### 密度函数

连续性随机变量充满某个区间,若恰好在[a,b]内取值,则随机变量$a \leq b,P(a \leq x \leq b) = 1$

\[
\int_b^a p(x) \, dx = P(a < x < b)
\]

> 函数上任意一点的取值都为零(概率为0的事也有可能发生),区间外的一点取值都为零

### 累积分布函数

\[
F(x)= p(X \leq x) = \int^x_{-\infty} p(t) \, dt \\[10pt]
\int^{+\infty}_{-\infty}p(t) \, dt = 1
\]

## 常用分布

### 二项分布

X为n重伯努利实验的成功次数,记为A,P为时间A发生的概率

\[
P(A) = p,X \sim b(n,p) \\[10pt]
P(x = k) = C_n^k p^k(1-p)^{n-k} \\[10pt]
\sum^n_{k=0} C_n^k p^k(1-p)^{n-k} = [p + (1-p)]^n = 1
\]

### 泊松分布

\[
记为X \sim P(\lambda) \\[10pt]
P(X=k) = \frac{\lambda^k}{k!} \, e^{-\lambda}
\]

### 超几何分布

\[
记为X \sim h(n,N,M) \\[10pt]
P(X = K) = \frac{C_M^k C_{N-M}^{N-k}}{C_N^n}
\]

### 几何分布

\[
X为时间A首次出现的试验次数 \\[10pt]
X \sim Ge(p) \\[10pt]
P(X = k) = (1-p)^{k-1}p \\[10pt]
\]

> 几何分布具有无记忆性,上一次的概率不会影响下一次的概率($P(x > m+n | x > m) = P(x > m)$)

### 正态分布

\[
密度函数: p(x) = \frac{1}{\sqrt{2\pi}\sigma}exp\{-\frac{(x-\mu)^2}{2 \sigma^2}\} \qquad \begin{cases}
\mu:& 控制位置,位置参数 \\[10pt]
\sigma: & 控制形状,尺度参数
\end{cases} \\[10pt]
记为:X \sim N(\mu,\sigma^2) \\[10pt]
分布函数: F(X) = \frac{1}{\sqrt{2\pi}\sigma} \int_{-\infty}^ x e^{-\frac{(t-\mu)^2}{2\sigma^2}} \, dt
\]

在现实中常把数据标准化为$\mu = 0, \sigma = 1$的标准正态分布,此时分布函数为$\phi(\mu)$,密度函数为$\varphi(\mu)$

#### 标准化

\[
若随机变量X \sim N(\mu,\sigma^2) \\[10pt]
则U = （X - \mu）/ \sigma \sim (0,1) \rightarrow 标准化 \\[10pt]
\phi(-\mu) = 1 - \phi(\mu) \\[10pt]
P(U > \mu) = 1 - \phi(\mu) \\[10pt]
P(a < U < b) = \phi(b) - \phi(a) \\[10pt]
P(|U| < c) = 2 \phi(c) - 1 \quad (c \geq 0)
\]

### 指数分布

常用于寿命,排队分布

\[
记为X \sim Exp(\lambda) \\[10pt]
p = \lambda e^{-\lambda x} \quad (x \geq 0)
\]

> 指数分布也具有无记忆性,上一次的概率不会影响下一次的概率($P(x > m+n | x > m) = P(x > m)$)

### 伽马分布

\[
\Gamma(a) = \int_0^{\infty} x^{a-1} e^{-x} \quad x常取x>1 \\[10pt]
\Gamma(n) = n! \qquad \Gamma(\frac{1}{2}) = \sqrt{\pi} \\[10pt]
记为X \sim Ga(a,\lambda) \begin{cases}
u > 0 \quad 形状参数 \\[10pt]
\lambda > 0 \quad 尺度参数
\end{cases} \\[10pt]
密度函数: p(x) = \frac{\lambda^n}{\Gamma(a)} x^{a-1} e^{-\lambda x} \quad (x \geq 0)
可表示为系统可以抵挡外来冲击,又遇到第k次时失败,第k次冲击来到的时间X服从伽马分布 \\[10pt]
Ga(1,\lambda) = Exp(\lambda) \quad (指数分布) \\[10pt]
Ga(\frac{n}{2},\frac{1}{2}) = X^2(n) \quad (卡方分布)
\]

### 贝塔分布

\[
贝塔函数记为B(a,b) \\[10pt]
B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)} \\[10pt]
贝塔分布记为X \sim B(a,b) \\[10pt]
P(X) = \frac{x^{a-1}(1-x)^{b-1}}{B(a,b)} \quad (0 < x < 1 且 a,b > 0) \\[10pt]
常用于不合格概率,机器维修率 \\[10pt]
(1,1) = U(0,1) \qquad 参数估计
\]
