# 导入必要的库
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

# 加载加州房价数据集
california = fetch_california_housing() # sklearn自带的数据集

# 将数据转换为 DataFrame pandas
california_df = pd.DataFrame(california.data, columns=california.feature_names)
california_df['MedHouseVal'] = california.target  # 添加目标变量（中位房价）（因变量，一般放在数据最后面）

# 数据预览和统计信息
print("前5行数据：")
print(california_df.head())
print("\n数据描述信息：")
print(california_df.describe())

# 通过热力图查看特征之间的相关性
plt.figure(figsize=(12, 8)) # 设置画布大小（1200x800）
sns.heatmap(california_df.corr(), annot=True, cmap='coolwarm', fmt=".2f") # corr()函数为数据的相关系数，annot为是否显示具体数据，cmap表示映射的方案，fmt为数据保留几位小数
plt.title("特征与房价（MedHouseVal）相关性热图") # 设置标题
plt.show()

# 设置多元线性回归方程的参数
features = ['MedInc', 'AveRooms', 'HouseAge']  # 中位收入、平均房间数、房屋年龄
X = california_df[features]  # 作为输入变量
y = california_df['MedHouseVal']  # 房价为目标变量

# 特征标准化处理
scaler = StandardScaler() # 创建对象
X_scaled = scaler.fit_transform(X) # 将数据缩放在[0,1]或者[-1,1]之间

# 建立并训练多元线性回归模型
model = LinearRegression() # 创建对象
model.fit(X_scaled, y) # 输入参数
predictions = model.predict(X_scaled) # 对模型进行预测

# 模型预测和评估
mse = mean_squared_error(y, predictions)  # 均方误差
rmse = np.sqrt(mse)  # 均方根误差
r2 = r2_score(y, predictions)  # 决定系数（拟合优度）

# 输出结果
print("模型系数（β）：", model.coef_)  # 每个特征的权重 [0.84115114 -0.06718067 0.21171013]
print("模型截距（β0）：", model.intercept_) # 2.068558169089147
print("均方误差 MSE：", mse) # 0.6496608827746702
print("均方根误差 RMSE：", rmse) # 0.8060154358166288
print("决定系数 R²：", r2) # 0.5121018839958533 <0.5为不合格模型 > 0.5 & < 0.7为一般 > 0.7 & < 1为优秀，这里的模型实际上不算好，只能说一般

# 绘制预测值和真实值的散点图
plt.figure(figsize=(8, 5)) # 设置画布（800x500）
plt.scatter(range(len(y)), y, label='实际价格', alpha=0.7) # 实际价格的散点图
plt.scatter(range(len(y)), predictions, label='预测价格', alpha=0.7) # 预测价格的散点图
plt.title("房价预测对比图") # 设置标题
plt.xlabel("样本编号") # 设置xlabel
plt.ylabel("价格") # 设置ylabel
plt.legend()
plt.grid(True) # 设置网格线
plt.show()