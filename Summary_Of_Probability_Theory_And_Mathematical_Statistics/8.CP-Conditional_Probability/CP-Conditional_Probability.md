# 条件概率

## 基础公式

### 乘法公式

\[
P(B|A) = \frac{P(AB)}{P(A)} \qquad P(A|B) = \frac{P(AB)}{P(B)} \\[10pt]
P(AB) = P(A)P(B|A) = P(B)P(A|B) \\[10pt]
P(A_1A_2\dots A_n) = P(A_1)P(A_2|A_1)P(A_3|A_1A_2)\dots P(A_n|A_1A_2\dots A_{n-1})
\]

### 全概率公式

\[
P(B) = P(AB) + P(\bar{A}B) = P(A)P(B|A) + P(\bar{A})P(B|\bar{A}) \\[10pt]
P(B) = P(A_1B) + P(A_2B) + \dots + P(A_nB) = \sum_{i=1}^nP(A_iB) = \sum_{i=1}^nP(A_i)P(B|A_i) \qquad 其中A_i \cap A_j = \emptyset \quad \cup A_i = \Omega
\]

### 贝叶斯公式

\[
P(A|B) = \frac{P(A)P(B|A)}{P(B)} = P(A|B) = \frac{P(A)P(B|A)}{\sum_{i=1}^nP(\Omega_i)P(B|\Omega_i)}
\]

## 波利亚模型(罐子模型)

设罐中有b个篮球,r个红球,每次抽取后加入同色球c个,异色球d个

|参数|放回抽样|不放回抽样|传染病模型|安全模型|
|:---:|:---:|:---:|:---:|:---:|
|c|0|-1|>0|0|
|d|0|0|0|>0|

### 不放回抽样模型(概率与顺序无关)

|参数|c = -1|d = 0|蓝 = b|红 = r|
|:---:|:---:|:---:|:---:|:---:|

\[
P(B_1) = \frac{b}{b+r} \\[10pt]
P(B_1R_2) = P(B_1)P(R_2|B_1) = \frac{b}{b+r}·\frac{r}{b+r-1} \\[10pt]
P(B_1R_2R_3) = P(B_1)P(R_2|B_1)P(R_3|B_1R_2) = \frac{b}{b+r}·\frac{r}{b+r-1}·\frac{r-1}{b+r-2} \\[10pt]
抽到1蓝2红的概率:P(B_1R_2R_3) + P(R_1B_2R_3) + P(R_1R_2B_3) = \frac{C_b^1C_r^2}{C_{b+r}^3} \\[10pt]
抽到m个篮球,n个红球的概率:P = \frac{C_b^mC_r^n}{C_{b+r}^{m+n}} \;超几何分布(抽次品问题)
\]

### 放回抽样模型(概率与顺序无关)

|参数|c = 0|d = 0|蓝 = b|红 = r|
|:---:|:---:|:---:|:---:|:---:|

\[
P(B_1) = \frac{b}{b+r} \\[10pt]
P(B_1R_2) = P(B_1)P(R_2|B_1) = \frac{b}{b+r}·\frac{r}{b+r} \\[10pt]
P(B_1R_2R_3) = P(B_1)P(R_2|B_1)P(R_3|B_1R_2) = \frac{b}{b+r}·\frac{r}{b+r}·\frac{r}{b+r} \\[10pt]
抽到1蓝2红的概率:P(B_1R_2R_3) + P(R_1B_2R_3) + P(R_1R_2B_3) = C_3^1p^1(1-p)^{3-1} \\[10pt]
抽到m个篮球,n个红球的概率:P = C_{m+n}^mp^m(1-p)^n \;二项分布
\]

### 传染病模型(概率与顺序无关)

|参数|c > 0|d = 0|正常人 = b|病人 = r|
|:---:|:---:|:---:|:---:|:---:|

\[
P(B_1) = \frac{b}{b+r} \\[10pt]
P(B_1R_2) = P(B_1)P(R_2|B_1) = \frac{b}{b+r}·\frac{r}{b+r+c} \\[10pt]
P(B_1R_2R_3) = P(B_1)P(R_2|B_1)P(R_3|B_1R_2) = \frac{b}{b+r}·\frac{r}{b+r+c}·\frac{r+c}{b+r+2c} \\[10pt]
抽到1正常人2病人的概率:P(B_1R_2R_3) + P(R_1B_2R_3) + P(R_1R_2B_3) = C_3^1\frac{\prod^0_{i=0}(b+ic)\prod^{2-1}_{j=0}(r+jc)}{\prod^{3-1}_{k=0}(b+r+kc)} \\[10pt]
抽到m个病人,n个正常人的概率:P = C^m_{m+n}\frac{\prod^{m-1}_{i=0}(r+ic)\prod^{n-1}_{j=0}(b+jc)}{\prod^{m+n-1}_{k=0}(b+r+kc)}
\]

### 安全模型(概率与顺序**有关**)

|参数|c = 0|d > 0|安全 = b|事故 = r|
|:---:|:---:|:---:|:---:|:---:|

抽到安全,事故概率升高;事故发生,抓紧安全,事故概率降低

\[
P(B_1R_2R_3) = P(B_1)P(R_2|B_1)P(R_3|B_1R_2) \\[10pt]
\]
