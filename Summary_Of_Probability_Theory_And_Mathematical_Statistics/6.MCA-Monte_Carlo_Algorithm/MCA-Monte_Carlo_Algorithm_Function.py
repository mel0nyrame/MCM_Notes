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

# 复杂函数的定积分计算

# 设置可视化风格
sns.set_style('whitegrid')  # 设置seaborn图表风格为白色网格，提高可读性
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 设置unicode_minus参数为False，用来正常显示负号