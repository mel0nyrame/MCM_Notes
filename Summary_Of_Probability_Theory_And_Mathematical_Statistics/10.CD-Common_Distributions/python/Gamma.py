# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/1 10:20
    @file  : gamma.py
"""

import math


class Gamma:
    """
    伽马分布类（Gamma Distribution）

    属性
    ------
    k : float
        形状参数（shape），k > 0
    lam : float
        尺度参数（rate，等于 1/scale），λ > 0

    方法
    ------
    pdf(x)         概率密度函数 f(x)
    expectation()  期望 E[X]
    variance()     方差 Var[X]
    std_dev()      标准差 Std[X]
    """

    def __init__(self, k: float, lam: float):
        """
        初始化伽马分布

        参数
        ------
        k : float
            形状参数 k > 0
        lam : float
            率参数 λ > 0
        """
        if k <= 0 or lam <= 0:
            raise ValueError("参数 k 和 λ 都必须大于 0")
        self.k = k
        self.lam = lam

    def pdf(self, x: float) -> float:
        """
        概率密度函数 PDF：f(x)

        .. math:


        参数
        ------
        x : float，必须大于 0

        返回
        ------
        float : f(x)
        """
        if x <= 0:
            return 0.0

        numerator = (self.lam ** self.k) * (x ** (self.k - 1)) * math.exp(-self.lam * x)
        denominator = math.gamma(self.k)
        return numerator / denominator

    def cdf(self, x: float, steps: int = 1000) -> float:
        """
        累积分布函数 CDF：P(X <= x)
        使用梯形积分法数值逼近

        参数
        ------
        x : float
            积分上限
        steps : int
            积分精度（越大越精确）

        返回
        ------
        float
            累计概率
        """
        if x <= 0:
            return 0.0

        a = 0
        b = x
        h = (b - a) / steps
        total = 0.5 * self.pdf(a) + 0.5 * self.pdf(b)

        for i in range(1, steps):
            t = a + i * h
            total += self.pdf(t)

        return h * total

    def expectation(self) -> float:
        """
        返回期望 E[X] = k / λ
        """
        return self.k / self.lam

    def variance(self) -> float:
        """
        返回方差 Var[X] = k / λ²
        """
        return self.k / (self.lam ** 2)

    def std_dev(self) -> float:
        """
        返回标准差 Std[X] = sqrt(Var[X])
        """
        return math.sqrt(self.variance())
