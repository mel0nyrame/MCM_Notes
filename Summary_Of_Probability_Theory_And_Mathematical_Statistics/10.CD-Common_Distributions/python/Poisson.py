# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/1 09:24
    @file  : poisson.py
"""

import math


class Poisson:
    """
    泊松分布类：使用纯数学方法计算泊松分布的概率特性

    Property
    ------
    lam : float
        单位时间（或单位面积、体积）内的平均事件发生次数 λ（lambda）

    Method
    ------
    pmf(k)
        计算概率质量函数 P(X = k)
    cdf(k)
        计算累积分布函数 P(X <= k)
    expectation()
        返回期望 E[X]
    variance()
        返回方差 Var[X]
    std_dev()
        返回标准差 Std[X]
    """

    def __init__(self, lam: float):
        """
        初始化泊松分布对象

        参数
        ------
        lam : float
            平均事件发生率 λ，必须大于 0
        """
        if lam <= 0:
            raise ValueError("λ (lambda) 必须为正数")
        self.lam = lam

    def _factorial(self, n: int):
        """
        计算阶乘 n!
        """
        if n < 0:
            raise ValueError("阶乘的输入必须是非负整数")
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def pmf(self, k: int):
        """
        概率质量函数 PMF

        参数
        ------
        k : int
            事件发生的次数

        返回
        ------
        float : P(X = k)
        """
        if k < 0:
            return 0.0
        return (self.lam ** k) * math.exp(-self.lam) / self._factorial(k)

    def cdf(self, k: int):
        """
        累积分布函数 CDF

        参数
        ------
        k : int
            事件发生次数的上限

        返回
        ------
        float : P(X <= k)
        """
        if k < 0:
            return 0.0
        total = 0.0
        for i in range(0, k + 1):
            total += self.pmf(i)
        return total

    def expectation(self):
        """
        返回期望 E[X] = λ
        """
        return self.lam

    def variance(self) -> float:
        """
        返回方差 Var[X] = λ
        """
        return self.lam

    def std_dev(self) -> float:
        """
        返回标准差 Std[X] = sqrt(λ)
        """
        return math.sqrt(self.lam)
