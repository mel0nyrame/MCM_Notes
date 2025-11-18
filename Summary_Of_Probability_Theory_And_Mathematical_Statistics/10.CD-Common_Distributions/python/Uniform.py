# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/1 10:15
    @file  : uniform.py
"""

import math

class Uniform:
    """
    连续型均匀分布类（Uniform Distribution）

    表示一个在 [a, b] 区间上均匀分布的随机变量

    属性
    ------
    a : float
        区间左端点
    b : float
        区间右端点（必须 a < b）

    方法
    ------
    pdf(x)         概率密度函数 f(x)
    cdf(x)         累积分布函数 P(X <= x)
    expectation()  期望 E[X]
    variance()     方差 Var[X]
    std_dev()      标准差 Std[X]
    """

    def __init__(self, a: float, b: float):
        """
        初始化均匀分布对象

        参数
        ------
        a : float
            区间起点
        b : float
            区间终点，需满足 b > a
        """
        if b <= a:
            raise ValueError("区间参数必须满足 b > a")
        self.a = a
        self.b = b

    def pdf(self, x: float) -> float:
        """
        概率密度函数 f(x)

        参数
        ------
        x : float
            输入的变量

        返回
        ------
        float : 对应的密度值
        """
        if self.a <= x <= self.b:
            return 1 / (self.b - self.a)
        else:
            return 0.0

    def cdf(self, x: float) -> float:
        """
        累积分布函数 P(X <= x)

        参数
        ------
        x : float

        返回
        ------
        float : 累计概率
        """
        if x < self.a:
            return 0.0
        elif x > self.b:
            return 1.0
        else:
            return (x - self.a) / (self.b - self.a)

    def expectation(self) -> float:
        """
        返回期望 E[X] = (a + b) / 2
        """
        return (self.a + self.b) / 2

    def variance(self) -> float:
        """
        返回方差 Var[X] = (b - a)^2 / 12
        """
        return ((self.b - self.a) ** 2) / 12

    def std_dev(self) -> float:
        """
        返回标准差 Std[X] = sqrt(Var[X])
        """
        return math.sqrt(self.variance())

