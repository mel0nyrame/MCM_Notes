from statistics import geometric_mean

import numpy as np
from pandas.core.ops import arithmetic_op

# 一致性检查

# 定义矩阵A (n阶矩阵数据)
A = np.array([[1, 5, 3, 2], [1/5, 1, 2, 1/3], [1/3, 1/2, 1, 2], [1/2, 3, 1/2, 1]])
# A = np.array([[1, 5, 3, 2], [1/5, 1, 1/2, 1/3], [1/3, 2, 1, 1/2], [1/2, 3, 2, 1]])

n = A.shape[0]  # 获取A的行

# 求出最大特征值以及对应的特征向量
eig_val, eig_vec = np.linalg.eig(A)  # eig_val是特征值， eig_vec是特征向量
Max_eig = max(eig_val)  # 求特征值的最大值

CI = (Max_eig - n) / (n - 1)
# RI表
RI = [0, 0.0001, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
# 注意哦，这里的RI最多支持 n = 15
# 这里n=2时，一定是一致矩阵，所以CI = 0，我们为了避免分母为0，将这里的第二个元素改为了很接近0的正数

CR = CI / RI[n]

print('一致性指标CI=', CI)
print('一致性比例CR=', CR)

if CR < 0.10:
    print('因为CR<0.10，所以该判断矩阵A的一致性可以接受!')
else:
    print('注意: CR >= 0.10，因此该判断矩阵A需要进行修改!')



def arithmetic_mean_method (array):
    # 算数平均法求权重

    # 定义判断矩阵A
    A = array

    # 计算每列的和
    ASum = np.sum(A, axis=0)

    # 获取A的行和列
    n, _ = A.shape

    # 归一化
    Stand_A = A / ASum

    # 各列相加到同一行
    ASumr = np.sum(Stand_A, axis=1)

    # 计算权重向量
    weights = ASumr / n

    return weights


def geometric_mean_method (array) :
    # 几何平均法求权重

    # 定义判断矩阵A
    A = array

    # 获取A的行和列
    n, _ = A.shape

    # 将A中每一行行元素相乘得到一列向量
    prod_A = np.prod(A, axis=1)

    # 将新的向量的每个分量开n次方等价求1/n次方
    prod_n_A = np.power(prod_A, 1 / n)

    # 归一化处理
    re_prod_A = prod_n_A / np.sum(prod_n_A)

    return re_prod_A


def eigenvalue_averaging_method (array) :
    # 特征值平均法求权重

    # 定义判断矩阵A
    A = np.array([[1, 5, 3, 2], [1 / 5, 1, 1 / 2, 1 / 3], [1 / 3, 2, 1, 1 / 2], [1 / 2, 3, 2, 1]])

    # 获取A的行和列
    n, _ = A.shape

    # 求出特征值和特征向量
    eig_values, eig_vectors = np.linalg.eig(A)

    # 找出最大特征值的索引
    max_index = np.argmax(eig_values)

    # 找出对应的特征向量
    max_vector = eig_vectors[:, max_index]

    # 对特征向量进行归一化处理，得到权重
    weights = max_vector / np.sum(max_vector)

    return weights
