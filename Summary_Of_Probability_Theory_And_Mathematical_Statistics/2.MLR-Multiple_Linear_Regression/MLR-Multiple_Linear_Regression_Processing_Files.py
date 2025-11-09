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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

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


# 1. 定义目标变量 (Target)
#    从训练集(df)中提取 'FloodProbability' 这一列
#    这是我们的模型需要学习和预测的目标
target=df['FloodProbability']

# 2. 定义训练特征 (Features)
#    从训练集(df)中移除 'id' 列和目标列 'FloodProbability'
#    axis=1 表示我们要删除的是“列”
#    剩下的所有列都作为模型的输入特征
features=df.drop(['id','FloodProbability'],axis=1)

# 3. 定义测试特征 (Test Features)
#    从测试集(test_df)中移除 'id' 列
#    注意：测试集（test_df）本身就没有 'FloodProbability' 列
#    这里要确保测试集的特征处理方式和训练集保持一致（即只移除 'id'）
features_test=test_df.drop(['id'], axis=1)

# 定义一个阈值
threshold = 11 # 定义一个阈值，用于后续的分段特征转换

# 定义一个函数来应用分段转换
def segmented_transform(feature, threshold):
    # np.where(条件, x, y) 满足条件返回x，否则返回y
    # 线性部分：如果特征值小于阈值，则保留原值，否则设为0
    linear_segment = np.where(feature < threshold, feature, 0)
    # 非线性部分：如果特征值大于等于阈值，则保留原值，否则设为0
    nonlinear_segment = np.where(feature >= threshold, feature, 0)

    # 组合转换：线性部分不变，非线性部分应用一个二阶和三阶的多项式组合
    transformed_feature = linear_segment + 0.5 * nonlinear_segment ** 2 + 0.5 * nonlinear_segment ** 3

    return transformed_feature # 返回转换后的新特征值


# --- 应用特征转换 ---

# 遍历训练集(features)的每一列（即每个特征）
for col in features.columns:
    # 对该列应用上面定义的分段转换函数
    features[col] = segmented_transform(features[col], threshold)

# 遍历测试集(features_test)的每一列
for col in features_test.columns:
    # 对测试集也应用完全相同的分段转换，确保数据一致性
    features_test[col] = segmented_transform(features_test[col], threshold)

# --- 模型构建与训练 ---

# 创建一个样条变换器（SplineTransformer）实例
# 样条变换是一种非线性变换，可以捕获更复杂的数据关系
spline_transformer = SplineTransformer(n_knots=6, degree=6, knots='uniform',include_bias=False)
# n_knots=6: 设置6个节点（控制点）
# degree=6: 设置样条的阶数为6
# knots='uniform': 节点在特征范围内均匀分布
# include_bias=False: 不包含偏置项（截距项）

# 创建一个处理管道（Pipeline）
# make_pipeline 用于将多个处理步骤（变换器、模型）串联起来
# 这里的管道包含两步：1. 应用样条变换 2. 拟合线性回归
pipeline = make_pipeline(spline_transformer, LinearRegression())

# 划分训练集和验证集
# X_train, X_test 是特征， y_train, y_test 是目标（洪水概率）
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=73)
# test_size=0.2: 指定验证集占总数据的 20%
# random_state=73: 设置随机种子，确保每次划分的结果都一样，便于复现

# 拟合模型
# pipeline.fit() 会自动先对 X_train 进行样条变换，然后用变换后的数据训练线性回归模型
pipeline.fit(X_train, y_train)

# --- 模型评估与预测 ---

# 在验证集上进行预测
# pipeline.predict() 会自动先对 X_test 进行样条变换，再用模型预测
y_pred = pipeline.predict(X_test)

# 评估模型
# r2_score (R² 决定系数) 是回归模型常用的评价指标，越接近1越好
r2score = r2_score(y_test, y_pred) # 计算真实值(y_test)和预测值(y_pred)的 R² 分数
print(f"r2 score : {r2score}") # 打印 R² 分数

# 对真实的测试集(features_test)进行最终预测
y_pred_test=pipeline.predict(features_test)
print(y_pred_test) # 打印测试集的预测结果

# --- 生成提交文件 ---

# 创建一个用于提交的 DataFrame
submission_df = pd.DataFrame({
    'id': test_df['id'], # 包含原始测试集中的 'id' 列
    'FloodProbability': y_pred_test # 包含模型预测出的 'FloodProbability' 列
})

# 将提交数据保存为 CSV 文件
# './output/submission.csv' 是指定的保存路径和文件名
# index=False 表示在保存时不需要写入 DataFrame 的行索引
submission_df.to_csv('./output/submission.csv', index=False)

print("Submission file created successfully.") # 打印提示信息：提交文件创建成功

# 读取刚保存的提交文件，通常用于在 Jupyter Notebook 中检查文件内容
pd.read_csv('./output/submission.csv')