# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/1 09:40
    @file  : norm.py
"""

import math


class Norm:
    r"""
    正态分布类（Normal Distribution）

        用于表示连续型概率分布，常用于自然界与社会现象建模。

        .. math::

        f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2 \sigma^2}\right)

    属性
    ------
    mu : float
        均值（期望），决定曲线中心位置
    sigma : float
        标准差，决定曲线宽度，必须 > 0

    方法
    ------
    pdf(x)
        返回概率密度函数 f(x)
    cdf(x)
        返回累积分布函数 P(X <= x)
    expectation()
        返回期望 E[X]
    variance()
        返回方差 Var[X]
    std_dev()
        返回标准差 Std[X]
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        """
        初始化正态分布

        参数
        ------
        mu : float
            分布均值（μ）
        sigma : float
            分布标准差（σ），必须为正
        """
        if sigma <= 0:
            raise ValueError("标准差 sigma 必须为正数")
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x: float) -> float:
        """
        概率密度函数 PDF：f(x)

        参数
        ------
        x : float
            任意实数

        返回
        ------
        float : 对应的概率密度值
        """
        coeff = 1 / (self.sigma * math.sqrt(2 * math.pi))
        exponent = -((x - self.mu) ** 2) / (2 * self.sigma ** 2)
        return coeff * math.exp(exponent)

    def cdf(self, x: float) -> float:
        """
        累积分布函数 CDF：P(X <= x)

        使用误差函数 erf() 近似计算

        返回
        ------
        float : 累计概率值
        """
        z = (x - self.mu) / (self.sigma * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def expectation(self) -> float:
        """
        返回期望 E[X] = μ
        """
        return self.mu

    def variance(self) -> float:
        """
        返回方差 Var[X] = σ^2
        """
        return self.sigma ** 2

    def std_dev(self) -> float:
        """
        返回标准差 Std[X] = σ
        """
        return self.sigma
