# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/1 09:33
    @file  : HyperGeom.py
"""

import math


class HyperGeom:
    """
    超几何分布类：用于模拟从有限总体中不放回抽样的概率问题

    属性
    ------
    N : int 总体总数
    
    K : int 总体中感兴趣的元素数量（“成功”元素）
    
    n : int 抽样数量（样本容量）

    方法
    ------
    pmf(k)        计算 P(X = k)，即抽中 k 个成功元素的概率
    
    cdf(k)        计算 P(X <= k)，即抽中 ≤k 个成功元素的概率
    
    expectation() 返回期望 E[X]
    
    variance()    返回方差 Var[X]
    
    std_dev()     返回标准差 Std[X]
    """

    def __init__(self, N: int, M: int, n: int):
        """
        初始化超几何分布参数

        参数
        ------
        N : int
            总体总数（必须 >= 1）
        M : int
            总体中成功的元素数量
        n : int
            抽样的数量

        约束
        ------
        0 <= K <= N
        0 <= n <= N
        """
        if not (0 <= M <= N):
            raise ValueError("K 必须满足 0 <= M <= N")
        if not (0 <= n <= N):
            raise ValueError("n 必须满足 0 <= n <= N")

        self.N = N
        self.M = M
        self.n = n


    def pmf(self, k: int) -> float:
        """
        概率质量函数：计算 P(X = k)

        参数
        ------
        k : int
            抽样中成功元素的个数

        返回
        ------
        float : P(X = k)
        """
        if k < 0 or k > self.n or k > self.M or self.n - k > self.N - self.M:
            return 0.0

        # P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
        numerator = math.comb(self.M, k) * math.comb(self.N - self.M, self.n - k)
        denominator = math.comb(self.N, self.n)
        return numerator / denominator

    def cdf(self, k: int) -> float:
        """
        累积分布函数：计算 P(X <= k)

        参数
        ------
        k : int
            抽中成功元素的最大数量

        返回
        ------
        float : P(X <= k)
        """
        total = 0.0
        lower = max(0, self.n + self.M - self.N)  # 可抽到的最小成功数
        upper = min(k, self.n, self.M)  # 上限由 k、本身 K 和 n 决定
        for i in range(lower, upper + 1):
            total += self.pmf(i)
        return total

    def expectation(self) -> float:
        """
        返回期望 E[X] = n * (K / N)
        """
        return self.n * (self.M / self.N)

    def variance(self) -> float:
        """
        返回方差 Var[X] = n * (K / N) * (1 - K / N) * (N - n) / (N - 1)
        """
        return (self.n * (self.M / self.N) *
                (1 - self.M / self.N) *
                (self.N - self.n) / (self.N - 1))

    def std_dev(self) -> float:
        """
        返回标准差 Std[X] = sqrt(Var[X])
        """
        return math.sqrt(self.variance())
