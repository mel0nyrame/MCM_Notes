# 导入必要的库
import os # 系统操作库
import pandas as pd  # 数据分析处理库
import numpy as np   # 数学计算库
import matplotlib.pyplot as plt  # 绘图工具
import seaborn as sns  # 更好看的可视化库
from sklearn.datasets import fetch_california_housing  # 加载加州房价数据集（系统自带的数据集）
from sklearn.linear_model import LinearRegression  # 多元线性回归模型 （同一元线性回归）
from sklearn.preprocessing import StandardScaler  # 标准化处理 （数据的标准化处理，让数据缩放到[0,1]之间）

from sklearn.metrics import mean_squared_error, r2_score  # 模型评价指标（MSE,R^2）
from sklearn.model_selection import cross_val_score  # 交叉验证模块 （验证模型是否有用）

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 从input文件夹读入数据
# 读取与洪水相关变量和发生概率
for dirname, _, filenames in os.walk('./input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# sample_submission,csv为提交数据，先训练出模型然后预测提交数据内编号的洪水发生率
df=pd.read_csv('./input/train.csv') # 训练数据
test_df = pd.read_csv('./input/test.csv') # 测试数据

# 展示数据信息
print(df.describe().T) # 输出训练数据的信息并转置
print(df.head()) # 输出训练数据的前五行

print(len(df)) # 训练数据的样本数
print(df.columns) # 训练数据的特征名称（列名）
print(test_df.columns) # 测试数据的特征名称（列名）

# 三种绘制频率图的方法

# --- 使用 Pandas 自带的 hist (直方图) ---
# .plot(kind='hist', ...) 是 Pandas DataFrame/Series 提供的快捷绘图方法
# kind='hist' 指定了图表类型为直方图
# bins=100 将数据范围（从最小值到最大值）分割成 100 个等宽的“桶”或“区间”
df['FloodProbability'].plot(kind='hist', bins=100)

# --- 设置图表元素 ---
plt.xlabel('FloodProbability') # 设置 X 轴的标签文字
plt.ylabel('Frequency (Count)') # 设置 Y 轴的标签文字（这里是频数/计数）
plt.title('Histogram of Flood Probability') # 设置图表的总标题
plt.show() # 显示绘制好的图表

# --- 使用 Seaborn 库的 histplot (直方图) ---
# sns.histplot() 是 Seaborn 中用于绘制直方图的函数
# 第一个参数 df['FloodProbability'] 是要绘制的数据
# bins=100 同样是指定分桶的数量
# kde=True 是一个非常有用的参数，它会在直方图上叠加一条“核密度估计”曲线
#         （Kernel Density Estimate），这条曲线是数据分布的平滑估计
sns.histplot(df['FloodProbability'], bins=100, kde=True)

plt.show() # 显示图表 (Seaborn 底层也依赖 Matplotlib)

# --- 使用 value_counts 和 sort_index (排序后的线图) ---
# --- 准备数据 ---
# 1. 统计 FloodProbability 列中，每个唯一的概率值出现了多少次
counts = df['FloodProbability'].value_counts()

# 2. .value_counts() 默认按“计数值”(Frequency) 降序排列
#    我们使用 .sort_index() 将其改为按“索引”(FloodProbability的值) 升序排列
#    这样 X 轴才能从小到大，绘制出正确的分布形状
counts_sorted = counts.sort_index()

# --- 绘图 ---
# 3. 对排序后的 Series 直接调用 .plot()，默认绘制线图 (line plot)
counts_sorted.plot()

# --- 设置图表元素 ---
plt.xlabel('FloodProbability') # X 轴是“概率值”
plt.ylabel('Frequency (Count)') # Y 轴是该概率值出现的“次数”
plt.title('Distribution of Flood Probability') # 设置标题
plt.show() # 显示图表

# 用图反应MonsoonIntensity和FloodProbability的关系，曲线波动越大，那么自变量对因变量的影响就越大，就越有可能是关键变量
# 剩下的变量以此类推
sns.catplot(x='InadequatePlanning',y='FloodProbability',data=df, kind='point')
plt.show()
