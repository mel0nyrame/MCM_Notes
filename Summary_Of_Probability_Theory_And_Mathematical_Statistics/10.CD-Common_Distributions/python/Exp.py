# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/1 10:16
    @file  : exp.py
"""

import math

class Exp:
    """
    指数分布类（Exponential Distribution）

    属性
    ------
    lam : float
        事件发生的速率参数 λ（lambda），必须大于 0

    方法
    ------
    pdf(x)         概率密度函数 f(x)
    cdf(x)         累积分布函数 P(X <= x)
    expectation()  返回期望 E[X]
    variance()     返回方差 Var[X]
    std_dev()      返回标准差 Std[X]
    """

    def __init__(self, lam: float):
        """
        初始化指数分布

        参数
        ------
        lam : float
            速率参数 λ（lambda），必须大于 0
        """
        if lam <= 0:
            raise ValueError("λ (lambda) 必须为正数")
        self.lam = lam

    def pdf(self, x: float) -> float:
        """
        概率密度函数 PDF：f(x)

        参数
        ------
        x : float
            自变量 x ≥ 0

        返回
        ------
        float : 密度值 f(x)
        """
        if x < 0:
            return 0.0
        return self.lam * math.exp(-self.lam * x)

    def cdf(self, x: float) -> float:
        """
        累积分布函数 CDF：P(X <= x)

        参数
        ------
        x : float

        返回
        ------
        float : 累积概率 P(X <= x)
        """
        if x < 0:
            return 0.0
        return 1 - math.exp(-self.lam * x)

    def expectation(self) -> float:
        """
        返回期望 E[X] = 1 / λ
        """
        return 1 / self.lam

    def variance(self) -> float:
        """
        返回方差 Var[X] = 1 / λ²
        """
        return 1 / (self.lam ** 2)

    def std_dev(self) -> float:
        """
        返回标准差 Std[X] = 1 / λ
        """
        return 1 / self.lam

