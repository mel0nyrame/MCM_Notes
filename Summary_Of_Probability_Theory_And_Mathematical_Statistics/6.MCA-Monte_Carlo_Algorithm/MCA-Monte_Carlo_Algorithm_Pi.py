# 数据处理和科学计算库
import numpy as np  # 导入numpy库，用于进行科学计算
import pandas as pd  # 导入pandas库，用于处理数据表
from datetime import datetime, timedelta  # 导入日期时间模块，用于处理时间序列数据

# 可视化库
import matplotlib.pyplot as plt  # 导入matplotlib库，用于可视化数据
import seaborn as sns  # 导入seaborn库，相比于matplotlib库有更多的函数，能够处理一些较为复杂的图
from scipy.optimize import curve_fit  # 导入curve_fit函数，用于非线性曲线拟合和参数估计

# 机器学习库
from sklearn.model_selection import train_test_split  # 导入sklearn(机器学习)库，用于分割训练数据和测试数据(一般80%用于训练,20%用于检验模型)
from sklearn.linear_model import LinearRegression  # 导入线性回归模型库
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # 导入三个评价指标（MSE.R^2,MAE）
from sklearn.model_selection import KFold  # 导入KFold交叉验证工具，用于将数据集分成K个子集进行交叉验证，评估模型稳定性

# 统计分析库
import statsmodels.api as sm  # 统计分析库，提供统计模型和推断工具
from scipy import stats  # 导入scipy的统计模块，提供概率分布和统计检验函数
from statsmodels.tsa.arima.model import ARIMA  # 导入ARIMA模型，用于时间序列预测和分析
from statsmodels.tsa.stattools import adfuller  # 导入ADF检验函数，用于检验时间序列的平稳性
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # 导入绘制自相关和偏自相关图的函数
from statsmodels.stats.diagnostic import acorr_ljungbox  # 导入Ljung-Box检验，用于检测时间序列的自相关性

# 忽略警告
import warnings
warnings.filterwarnings('ignore')  # 忽略运行中产生的警告信息，使输出更清晰

# 计算圆周率

# 设置可视化风格
sns.set_style('whitegrid')  # 设置seaborn图表风格为白色网格，提高可读性
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 设置unicode_minus参数为False，用来正常显示负号

# 设置种子
np.random.seed(42)

# 模拟数据
def estimate_pi_single(n):
    """
    使用蒙特卡洛方法估算π值

    参数:
        n: 随机点的数量

    返回:
        pi_estimate: π的估计值
        x, y: 随机点坐标
        inside_circle: 布尔数组，标记点是否在圆内
    """
    # 生成随机点坐标
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)

    # 判断点是否在单位圆内
    inside_circle = (x ** 2 + y ** 2) <= 1

    # 计算π的估计值
    pi_estimate = 4 * np.sum(inside_circle) / n

    return pi_estimate, x, y, inside_circle

# 单次估计圆周率
n_points = 10000
pi_est, x_coords, y_coords, is_inside = estimate_pi_single(n_points)

print(f"使用 {n_points} 个随机点")
print(f"π的估计值: {pi_est:.6f}")
print(f"真实π值: {np.pi:.6f}")
print(f"绝对误差: {abs(pi_est - np.pi):.6f}")
print(f"相对误差: {abs(pi_est - np.pi)/np.pi*100:.3f}%")

# 可视化随机点分布
plt.figure(figsize=(12, 5))
# 子图1：显示所有点
plt.subplot(1, 2, 1)
plt.scatter(x_coords[is_inside], y_coords[is_inside], c='red', s=1, alpha=0.6, label='圆内点')
plt.scatter(x_coords[~is_inside], y_coords[~is_inside], c='blue', s=1, alpha=0.6, label='圆外点')

# 绘制单位圆
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='单位圆')

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend()
plt.title(f'随机点分布 (n={n_points})')
plt.xlabel('x')
plt.ylabel('y')

# 子图2：部分放大显示
plt.subplot(1, 2, 2)
n_show = 1000  # 只显示前1000个点，避免图像过于密集
plt.scatter(x_coords[:n_show][is_inside[:n_show]], y_coords[:n_show][is_inside[:n_show]],
           c='red', s=10, alpha=0.7, label='圆内点')
plt.scatter(x_coords[:n_show][~is_inside[:n_show]], y_coords[:n_show][~is_inside[:n_show]],
           c='blue', s=10, alpha=0.7, label='圆外点')
plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='单位圆')

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend()
plt.title(f'放大显示 (前{n_show}个点)')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# 收敛性分析
def analyze_convergence(max_n=50000, step=1000):
    """分析π估计的收敛过程"""
    n_values = range(step, max_n + 1, step)
    pi_estimates = []
    errors = []

    # 生成一次性的大量随机点
    x_all = np.random.uniform(-1, 1, max_n)
    y_all = np.random.uniform(-1, 1, max_n)
    inside_all = (x_all ** 2 + y_all ** 2) <= 1

    for n in n_values:
        # 使用前n个点计算π
        pi_est = 4 * np.sum(inside_all[:n]) / n
        pi_estimates.append(pi_est)
        errors.append(abs(pi_est - np.pi))

    return n_values, pi_estimates, errors


n_vals, pi_ests, errs = analyze_convergence()

# 绘制收敛性曲线

plt.figure(figsize=(15, 5))

# 子图1：π估计值随样本数变化
plt.subplot(1, 3, 1)
plt.plot(n_vals, pi_ests, 'b-', linewidth=1, alpha=0.7, label='π估计值')
plt.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label=f'真实值 π={np.pi:.6f}')
plt.xlabel('样本数量')
plt.ylabel('π估计值')
plt.title('π估计值收敛过程')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：绝对误差随样本数变化
plt.subplot(1, 3, 2)
plt.plot(n_vals, errs, 'g-', linewidth=1, alpha=0.7, label='绝对误差')
plt.plot(n_vals, 1/np.sqrt(n_vals), 'r--', linewidth=2, alpha=0.7, label='理论收敛速度 O(1/√n)')
plt.xlabel('样本数量')
plt.ylabel('绝对误差')
plt.title('误差收敛分析')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.xscale('log')

# 子图3：相对误差百分比
plt.subplot(1, 3, 3)
relative_errors = [err/np.pi*100 for err in errs]
plt.plot(n_vals, relative_errors, 'm-', linewidth=1, alpha=0.7)
plt.xlabel('样本数量')
plt.ylabel('相对误差 (%)')
plt.title('相对误差变化')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 输出统计信息
print(f"最终估计值 (n={n_vals[-1]}): {pi_ests[-1]:.6f}")
print(f"最终相对误差: {relative_errors[-1]:.3f}%")

# 1.4 多次独立试验的统计分析
print("\n1.4 多次独立试验的统计分析")

def multiple_trials(n_points, n_trials=100):
    """进行多次独立的π估计试验"""
    estimates = []
    for _ in range(n_trials):
        pi_est, _, _, _ = estimate_pi_single(n_points)
        estimates.append(pi_est)
    return np.array(estimates)

n_test = 10000
n_trials = 100
trial_results = multiple_trials(n_test, n_trials)

# 多次独立试验的统计分析
print(f"进行 {n_trials} 次独立试验，每次使用 {n_test} 个点")
print(f"π估计值的均值: {np.mean(trial_results):.6f}")
print(f"π估计值的标准差: {np.std(trial_results):.6f}")
print(f"理论标准误差: {np.sqrt(np.pi*(4-np.pi)/n_test):.6f}")

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(trial_results, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.pi, color='red', linestyle='--', linewidth=2, label=f'真实值 π={np.pi:.6f}')
plt.axvline(np.mean(trial_results), color='green', linestyle='-', linewidth=2,
           label=f'样本均值={np.mean(trial_results):.6f}')
plt.xlabel('π估计值')
plt.ylabel('概率密度')
plt.title(f'{n_trials}次独立试验结果分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()