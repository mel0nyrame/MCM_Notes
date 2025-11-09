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

# 残差分析
residuals = y - predictions  # 残差 = 实际值 - 预测值
plt.figure(figsize=(8, 5)) # 设置画布（800x500）
plt.scatter(predictions, residuals, alpha=0.6) # 绘制预测值和真实值的残差图
plt.hlines(y=0, xmin=predictions.min(), xmax=predictions.max(), colors='red') # 添加红色水平参考线
plt.title("残差图（Residual Plot）") # 设置标题
plt.xlabel("预测值") # 设置xlabel
plt.ylabel("残差") # 设置ylabel
plt.grid(True) # 设置网格线
plt.show()

# 交叉验证
cross_val_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')  # 5折交叉验证，意思是将数据分成五份，每次将其中一份作为测试集，其他作为数据集，循环五次，反复验证
print("5折交叉验证 R² 分数：", cross_val_scores)
print("平均交叉验证得分：", cross_val_scores.mean())

# 误差分布分析 检查误差分布是否满足正态分布曲线，越接近正态分布模型越好
plt.figure(figsize=(8, 5)) # 设置画布（800x500）
sns.histplot(residuals, kde=True, bins=30, color='purple') # 绘制残差直方图并叠加核密度曲线
plt.title("残差分布图") # 设置标题
plt.xlabel("残差值") # 设置xlabel
plt.ylabel("频率") # 设置ylabel
plt.grid(True) # 设置网格线
plt.show()

# 保存回归分析结果到CSV文件

mse = mean_squared_error(y, predictions)  # 均方误差
rmse = np.sqrt(mse)  # 均方根误差
r2 = r2_score(y, predictions)  # 决定系数（拟合优度）

results_df = pd.DataFrame({
    '特征名称': features,
    '回归系数': model.coef_
})

summary_df = pd.DataFrame({
    '项': ['截距', 'MSE', 'RMSE', 'R²', '交叉验证均值R²'],
    '值': [model.intercept_, mse, rmse, r2, cross_val_scores.mean()]
})

full_results_df = pd.concat([results_df, pd.DataFrame([{}]), summary_df], ignore_index=True)
full_results_df.to_csv("regression_results.csv", index=False, encoding='utf-8-sig')
print("✅ 回归分析结果已保存至 regression_results.csv")

# ✅ 可选扩展任务 5：保存预测值和残差
pred_df = pd.DataFrame({
    '实际房价': y,
    '预测房价': predictions,
    '残差': residuals
})
pred_df.to_csv("predictions_residuals.csv", index=False, encoding='utf-8-sig')
print("✅ 预测值与残差数据已保存至 predictions_residuals.csv")
