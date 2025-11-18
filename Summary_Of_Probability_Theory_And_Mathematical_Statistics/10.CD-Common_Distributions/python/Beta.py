# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/1 10:22
    @file  : beta.py
"""

import math


class Beta:
    """
    贝塔分布类（Beta Distribution）

    属性
    ------
    alpha : float
        形状参数 α > 0
    beta : float
        形状参数 β > 0

    方法
    ------
    pdf(x)         概率密度函数 f(x)
    expectation()  期望 E[X]
    variance()     方差 Var[X]
    std_dev()      标准差 Std[X]
    """

    def __init__(self, alpha: float, beta: float):
        """
        初始化贝塔分布对象

        参数
        ------
        alpha : float
            α > 0
        beta : float
            β > 0
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha 和 beta 必须都大于 0")
        self.alpha = alpha
        self.beta = beta

    def _beta_function(self, alpha: float, beta: float) -> float:
        """
        计算贝塔函数 B(α, β) = Γ(α)Γ(β)/Γ(α+β)
        """
        return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

    def pdf(self, x: float) -> float:
        r"""
        概率密度函数 PDF

            math::
                \frac(\Gamma(\alpha+\beta))/(\Gamma(\alpha)+\Gamma(\beta))x^{\alpha-1}(1-x)^{\beta-1}

        参数
        ------
        x : float，定义在 (0, 1)

        返回
        ------
        float : f(x)
        """
        if x <= 0 or x >= 1:
            return 0.0

        coeff = 1 / self._beta_function(self.alpha, self.beta)
        return coeff * (x ** (self.alpha - 1)) * ((1 - x) ** (self.beta - 1))

    def cdf(self, x: float, steps: int = 1000) -> float:
        """
        累积分布函数 CDF：P(X <= x)
        使用梯形法数值逼近

        参数
        ------
        x : float ∈ [0, 1]
        steps : int
            积分精度（越大越精确）

        返回
        ------
        float : 累计概率
        """
        if x <= 0:
            return 0.0
        elif x >= 1:
            return 1.0

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
        返回期望 E[X] = α / (α + β)
        """
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """
        返回方差 Var[X] = αβ / [(α+β)^2 (α+β+1)]
        """
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def std_dev(self) -> float:
        """
        返回标准差 Std[X] = sqrt(Var[X])
        """
        return math.sqrt(self.variance())

