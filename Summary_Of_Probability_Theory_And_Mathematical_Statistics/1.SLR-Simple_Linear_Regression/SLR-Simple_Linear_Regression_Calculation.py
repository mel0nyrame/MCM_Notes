# 数据处理和科学计算库
import numpy as np # 导入numpy库，用于进行科学计算
import pandas as pd # 导入pandas库，用于处理数据表

# 可视化库
import matplotlib.pyplot as plt # 导入matplotlib库，用于可视化数据
import seaborn as sns # 导入seaborn库，相比于matplotlib库有更多的函数，能够处理一些较为复杂的图

# 机器学习库
from sklearn.model_selection import train_test_split # 导入sklearn(机器学习)库，用于分割训练数据和测试数据(一般80%用于训练,20%用于检验模型)
from sklearn.linear_model import LinearRegression # 导入线性回归模型库
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # 导入三个评价指标（MSE,R^2,MAE）

# 统计分析库
import statsmodels.api as sm # 统计分析库
from scipy import stats # 统计分析库

# 一元线性回归计算

# 设置可视化风格
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 沿用例子

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 生成100个学生的学习时间（小时）
study_hours = np.random.uniform(1, 10, 100).reshape(-1, 1)

# 生成对应的考试成绩（分数），假设基础分是30，每小时学习增加7分，并加入一些随机噪声
exam_scores = 30 + 7 * study_hours + np.random.normal(0, 5, 100).reshape(-1, 1)

# 将数据转换为Pandas DataFrame，方便处理
data = pd.DataFrame({'Study_Hours': study_hours.flatten(), 'Exam_Scores': exam_scores.flatten()})

# 已封装好的函数
def run_linear_regression_analysis(data, feature_col, target_col):
    """
    执行一元线性回归分析的完整流程模板。

    参数:
    data (pd.DataFrame): 包含数据的DataFrame。
    feature_col (str): 自变量列名。
    target_col (str): 因变量列名。

    返回:
    dict: 包含模型、评估指标和预测结果的字典。
    """
    # 1. 数据准备和划分
    X = data[[feature_col]]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. 模型训练
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. 预测与评估
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 4. 打印结果
    print(f'--- 分析报告: {feature_col} vs {target_col} ---')
    print(f'回归方程: {target_col} = {model.intercept_:.2f} + {model.coef_[0]:.2f} * {feature_col}')
    print(f'R-squared (R²): {r2:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

    # 5. 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='实际值')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='预测回归线')
    plt.title(f'{feature_col} 与 {target_col} 的回归分析')
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.legend()
    plt.show()

    return {
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'predictions_df': pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    }


# 使用模板函数
results = run_linear_regression_analysis(data, 'Study_Hours', 'Exam_Scores')