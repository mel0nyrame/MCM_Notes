# 数据处理和科学计算库
import numpy as np # 导入numpy库，用于进行科学计算
import pandas as pd # 导入pandas库，用于处理数据表

# 可视化库
import matplotlib.pyplot as plt # 导入matplotlib库，用于可视化数据
import seaborn as sns # 导入seaborn库，相比于matplotlib库有更多的函数，能够处理一些较为复杂的图

# 机器学习库
from sklearn.model_selection import train_test_split # 导入sklearn(机器学习)库，用于分割训练数据和测试数据(一般80%用于训练,20%用于检验模型)
from sklearn.linear_model import LinearRegression # 导入线性回归模型库
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # 导入三个评价指标（MSE.R^2,MAE）

# 统计分析库
import statsmodels.api as sm # 统计分析库
from scipy import stats # 统计分析库

# 数据生成与探索性分析

# 场景： 我们假设要研究“学习时间”与“考试成绩”之间的关系

# 设置可视化风格
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子以保证结果可复现
np.random.seed(114514)

# 生成100个学生的学习时间（小时）
study_hours = np.random.uniform(1, 10, 100).reshape(-1, 1)

# 生成对应的考试成绩（分数），假设基础分是30，每小时学习增加7分，并加入一些随机噪声
exam_scores = 30 + 7 * study_hours + np.random.normal(0, 5, 100).reshape(-1, 1)

# 将数据转换为Pandas DataFrame，方便处理
data = pd.DataFrame({'Study_Hours': study_hours.flatten(), 'Exam_Scores': exam_scores.flatten()})

# 数据的基本信息，例如中位数，百分数等
print("数据基本统计信息:")
print(data.describe())

# 在进行绘制之前需要做缺失值检查,就比如说表中的数据存不存在，是否是null等
# 若存在缺失值，可以使用均值填充法，讲缺失值用均值填充上；也可以使用删除缺失值，直接把整行数据删除了

# 数据可视化
plt.figure(figsize=(10, 6)) # 创建画布（分辨率为1000x600）
sns.scatterplot(x='Study_Hours', y='Exam_Scores', data=data, alpha=0.7) # 绘制散点图（x为学习时间，y为考试成，data为处理过后的pandas数据，alpha为点的大小）
plt.title('学习时间与考试成绩的关系', fontsize=15) # 设置标题和字体大小
plt.xlabel('学习时间 (小时)', fontsize=12) # 设置x轴的label和字体大小
plt.ylabel('考试成绩 (分)', fontsize=12) # 设置y轴的label和字体大小
plt.show()

# 计算相关系数
correlation = data['Study_Hours'].corr(data['Exam_Scores']) # 默认方法计算相关系数
print(f"学习时间与考试成绩的相关系数为: {correlation:.4f}") # 输出相关系数，保留四位小数；输出值接近于1，表示函数为强线性关系

# 定义自变量X和因变量y
X = data[['Study_Hours']]  # 自变量需要是2D数组
y = data['Exam_Scores']

# 将数据按80:20的比例划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 数据顺序会被打乱

print(f'训练集大小: {X_train.shape[0]}') # 80
print(f'测试集大小: {X_test.shape[0]}') # 20

# 训练模型

# 1. 创建线性回归模型实例
model = LinearRegression()

# 2. 使用训练数据来训练模型
model.fit(X_train, y_train)

# 3. 获取模型的参数
intercept = model.intercept_  # 截距 (beta_0)
slope = model.coef_[0]      # 斜率 (beta_1)

print(f'模型训练完成!')
print(f'截距 (beta_0): {intercept:.4f}')
print(f'斜率 (beta_1): {slope:.4f}')

print('回归方程为:')
print(f'Exam_Scores = {intercept:.2f} + {slope:.2f} * Study_Hours')

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 将预测结果和实际结果放在一个DataFrame里方便比较
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("测试集上的预测结果 vs 实际结果:")
print(predictions_df.head())

# 计算R² (决定系数)
r2 = r2_score(y_test, y_pred)
# 计算MSE (均方误差)
mse = mean_squared_error(y_test, y_pred)
# 计算RMSE (均方根误差)
rmse = np.sqrt(mse)
# 计算MAE (平均绝对误差)
mae = mean_absolute_error(y_test, y_pred)

print(f'--- 模型在测试集上的评估结果 ---')
print(f'R-squared (R²): {r2:.4f}') # 我们的模型可以解释约{r2:.4f}的考试成绩变化，这是一个非常高的值，说明学习时间这个变量对考试成绩有很强的解释力
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}') # 模型的预测值与真实值之间的平均误差大约是{mse:.4f}分，考虑到考试成绩的范围，这是一个相当小的误差
print(f'Mean Absolute Error (MAE): {mae:.4f}') # 模型的预测值与真实值的绝对偏差平均为{rmse:.4f}分，这个值通常比RMSE小，因为它对大误差不那么敏感

# 训练出来的模型和数据的可视化

plt.figure(figsize=(10, 6))

# 绘制原始数据散点图 (测试集)
plt.scatter(X_test, y_test, color='blue', label='实际成绩', alpha=0.7)

# 绘制回归线
plt.plot(X_test, y_pred, color='red', linewidth=2, label='回归线')

plt.title('线性回归拟合结果 (测试集)', fontsize=15)
plt.xlabel('学习时间 (小时)', fontsize=12)
plt.ylabel('考试成绩 (分)', fontsize=12)
plt.legend()
plt.show()

# 计算残差（残差 = 实际值 - 预测值）
residuals = y_test - y_pred

# 残差图是检验模型假设的重要工具，一个好的模型的残差应该看起来是随机分布的，没有明显的模式

# 残差点随机分布在y=0的水平线上下，没有形成明显的“喇叭口”或曲线形状。
# 这表明模型的线性假设和同方差性假设基本成立。

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--') # 添加一条y=0的参考线
plt.title('残差图', fontsize=15)
plt.xlabel('预测值', fontsize=12)
plt.ylabel('残差', fontsize=12)
plt.show()

# 假设一个新学生学习了6.5个小时，我们来预测他的成绩
new_study_hours = np.array([[6.5]])

# 使用模型进行预测
predicted_score = model.predict(new_study_hours)

print(f'预测：学习6.5小时的学生的考试成绩约为 {predicted_score[0]:.2f} 分。')