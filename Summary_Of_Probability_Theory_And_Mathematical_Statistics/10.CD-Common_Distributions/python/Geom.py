# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/1 09:38
    @file  : geom.py
"""

import math

class Geom:
    """
    几何分布类：表示第一次成功所需的试验次数（X ∈ {1, 2, 3, ...}）

    属性
    ------
    p : float
        每次伯努利试验成功的概率（0 < p <= 1）

    方法
    ------
    pmf(k)        返回 P(X = k)：第 k 次试验才成功的概率
    cdf(k)        返回 P(X <= k)：在前 k 次试验中至少成功一次的概率
    expectation() 返回期望 E[X]
    variance()    返回方差 Var[X]
    std_dev()     返回标准差 Std[X]
    """

    def __init__(self, p: float):
        """
        初始化几何分布对象

        参数
        ------
        p : float
            每次试验成功的概率，必须满足 0 < p <= 1
        """
        if not (0 < p <= 1):
            raise ValueError("成功概率 p 必须在 (0, 1] 区间内")
        self.p = p

    def pmf(self, k: int) -> float:
        """
        概率质量函数：P(X = k)

        参数
        ------
        k : int
            第 k 次才成功

        返回
        ------
        float : P(X = k)
        """
        if k < 1:
            return 0.0
        return (1 - self.p) ** (k - 1) * self.p

    def cdf(self, k: int) -> float:
        """
        累积分布函数：P(X <= k)

        参数
        ------
        k : int
            最大尝试次数

        返回
        ------
        float : P(X <= k)
        """
        if k < 1:
            return 0.0
        return 1 - (1 - self.p) ** k

    def expectation(self) -> float:
        """
        返回期望 E[X] = 1 / p
        """
        return 1 / self.p

    def variance(self) -> float:
        """
        返回方差 Var[X] = (1 - p) / p^2
        """
        return (1 - self.p) / (self.p ** 2)

    def std_dev(self) -> float:
        """
        返回标准差 Std[X] = sqrt(Var[X])
        """
        return math.sqrt(self.variance())

