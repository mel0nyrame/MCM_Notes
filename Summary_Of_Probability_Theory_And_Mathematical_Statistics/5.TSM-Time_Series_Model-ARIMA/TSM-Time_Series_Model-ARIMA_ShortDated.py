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

# 股票价格预测

# 设置可视化风格
sns.set_style('whitegrid')  # 设置seaborn图表风格为白色网格，提高可读性
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 设置unicode_minus参数为False，用来正常显示负号

# 数据生成
def generate_stock_data(start_date='2023-01-01', periods=180, initial_price=100):
    """
    生成模拟股票价格数据
    包含趋势、波动率和随机游走特征
    """
    np.random.seed(114514)  # 设置随机种子为114514，保证结果可重复

    # 创建日期索引
    dates = pd.date_range(start=start_date, periods=periods, freq='D')

    # 生成价格序列：随机游走 + 微弱趋势 + 波动率变化
    returns = np.random.normal(0.001, 0.02, periods)  # 生成日收益率，均值为0.1%，标准差为2%
    trend = np.linspace(0, 0.5, periods) / periods  # 生成微弱上升趋势项
    volatility = 1 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, periods))  # 生成波动率周期性变化项

    returns = returns + trend  # 将趋势项加入收益率
    returns = returns * volatility  # 乘以波动率调整项

    # 计算累积价格
    prices = [initial_price]  # 初始化价格列表，从初始价格开始
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))  # 根据收益率递推计算价格

    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,  # 日期列
        'price': prices,  # 价格列
        'returns': [0] + list(np.diff(np.log(prices)))  # 对数收益率列，第一个值为0
    })
    df.set_index('date', inplace=True)  # 将日期列设置为索引

    return df

# 数据预览
stock_data = generate_stock_data()  # 调用函数生成模拟股票数据
print(f"数据期间: {stock_data.index[0].strftime('%Y-%m-%d')} 至 {stock_data.index[-1].strftime('%Y-%m-%d')}")  # 打印数据时间范围
print(f"数据点数: {len(stock_data)}")  # 打印数据点数量
print(f"价格范围: {stock_data['price'].min():.2f} - {stock_data['price'].max():.2f}")  # 打印价格最小最大值

print("\n数据基本统计:")  # 打印基本统计信息标题
print(stock_data.describe())  # 显示数据的描述性统计（均值、标准差、分位数等）
print(stock_data.head())  # 显示数据前5行

# 数据可视化
# 创建综合图表
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 创建2×2的子图布局，设置图形大小

# 价格时序图
axes[0,0].plot(stock_data.index, stock_data['price'], color='steelblue', linewidth=1.5)  # 绘制价格曲线
axes[0,0].set_title('股票价格时序图', fontsize=14, fontweight='bold')  # 设置标题
axes[0,0].set_ylabel('价格 (元)')  # 设置Y轴标签
axes[0,0].grid(True, alpha=0.3)  # 显示网格线，设置透明度

# 收益率时序图
axes[0,1].plot(stock_data.index, stock_data['returns'], color='orange', linewidth=1)  # 绘制收益率曲线
axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)  # 添加零参考线
axes[0,1].set_title('日收益率时序图', fontsize=14, fontweight='bold')  # 设置标题
axes[0,1].set_ylabel('收益率')  # 设置Y轴标签
axes[0,1].grid(True, alpha=0.3)  # 显示网格线

# 价格分布直方图
axes[1,0].hist(stock_data['price'], bins=30, color='lightblue', alpha=0.7, edgecolor='black')  # 绘制价格分布
axes[1,0].set_title('价格分布直方图', fontsize=14, fontweight='bold')  # 设置标题
axes[1,0].set_xlabel('价格 (元)')  # 设置X轴标签
axes[1,0].set_ylabel('频数')  # 设置Y轴标签

# 收益率分布直方图
axes[1,1].hist(stock_data['returns'], bins=30, color='lightcoral', alpha=0.7, edgecolor='black')  # 绘制收益率分布
axes[1,1].set_title('收益率分布直方图', fontsize=14, fontweight='bold')  # 设置标题
axes[1,1].set_xlabel('收益率')  # 设置X轴标签
axes[1,1].set_ylabel('频数')  # 设置Y轴标签

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图形

# 对数据进行平稳性检验和差分
# ADF平稳性检验
def detailed_adf_test(timeseries, title="时间序列"):
    """
    详细的ADF平稳性检验
    返回布尔值表示是否平稳
    """
    result = adfuller(timeseries.dropna())  # 执行ADF检验，删除缺失值

    print(f'\n{title} ADF检验结果:')  # 打印检验标题
    print(f'ADF统计量: {result[0]:.6f}')  # 打印ADF统计量
    print(f'p值: {result[1]:.6f}')  # 打印p值
    print('临界值:')  # 打印临界值标题
    for key, value in result[4].items():
        print(f'  {key}: {value:.3f}')  # 打印各置信水平下的临界值

    if result[1] <= 0.05:  # 判断p值是否小于显著性水平0.05
        print(f'结论: {title}是平稳的 (p值 = {result[1]:.6f} < 0.05)')  # 平稳结论
        return True
    else:
        print(f'结论: {title}是非平稳的 (p值 = {result[1]:.6f} > 0.05)')  # 非平稳结论
        return False


# 检验原始价格序列
price_stationary = detailed_adf_test(stock_data['price'], "原始价格序列")  # 对价格序列进行ADF检验

# 检验收益率序列
returns_stationary = detailed_adf_test(stock_data['returns'], "收益率序列")  # 对收益率序列进行ADF检验

# 如果价格序列非平稳，进行差分
if not price_stationary:
    stock_data['price_diff'] = stock_data['price'].diff()  # 计算价格一阶差分
    diff_stationary = detailed_adf_test(stock_data['price_diff'], "价格一阶差分序列")  # 检验差分后序列的平稳性

# 可视化平稳性
fig, axes = plt.subplots(3, 1, figsize=(15, 12))  # 创建3行1列的子图布局

# 原始价格
axes[0].plot(stock_data['price'], color='blue', linewidth=1.5)  # 绘制原始价格序列
axes[0].set_title('原始价格序列', fontsize=14, fontweight='bold')  # 设置标题
axes[0].set_ylabel('价格')  # 设置Y轴标签
axes[0].grid(True, alpha=0.3)  # 显示网格线

# 收益率
axes[1].plot(stock_data['returns'], color='green', linewidth=1)  # 绘制收益率序列
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)  # 添加零参考线
axes[1].set_title('收益率序列 (平稳)', fontsize=14, fontweight='bold')  # 设置标题，标记为平稳
axes[1].set_ylabel('收益率')  # 设置Y轴标签
axes[1].grid(True, alpha=0.3)  # 显示网格线

# 价格差分
axes[2].plot(stock_data['price_diff'], color='orange', linewidth=1)  # 绘制价格差分序列
axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.7)  # 添加零参考线
axes[2].set_title('价格一阶差分序列', fontsize=14, fontweight='bold')  # 设置标题
axes[2].set_ylabel('价格差分')  # 设置Y轴标签
axes[2].set_xlabel('日期')  # 设置X轴标签
axes[2].grid(True, alpha=0.3)  # 显示网格线

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图形

# ACF和PACF分析
# 选择平稳序列进行分析（这里选择收益率序列）
analysis_data = stock_data['returns'].dropna()  # 选择收益率序列并删除缺失值

# 绘制ACF和PACF图
fig, axes = plt.subplots(2, 1, figsize=(15, 10))  # 创建2行1列的子图布局

# ACF图
plot_acf(analysis_data, ax=axes[0], lags=20, title='自相关函数 (ACF)')  # 绘制ACF图，显示20个滞后期
axes[0].set_xlabel('滞后期')  # 设置X轴标签
axes[0].grid(True, alpha=0.3)  # 显示网格线

# PACF图
plot_pacf(analysis_data, ax=axes[1], lags=20, title='偏自相关函数 (PACF)')  # 绘制PACF图，显示20个滞后期
axes[1].set_xlabel('滞后期')  # 设置X轴标签
axes[1].grid(True, alpha=0.3)  # 显示网格线

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图形


# 1.分析ACF和PACF的截尾特征来获取ARIMA的三个参数p,d,q
def analyze_acf_pacf(data, max_lags=10):
    """
    分析ACF和PACF的截尾特征，建议模型参数
    返回建议的AR阶数p和MA阶数q
    """
    from statsmodels.tsa.stattools import acf, pacf  # 导入计算ACF和PACF的函数

    # 计算ACF和PACF值
    acf_values = acf(data, nlags=max_lags, fft=False)  # 计算自相关系数
    pacf_values = pacf(data, nlags=max_lags)  # 计算偏自相关系数

    # 95%置信区间
    n = len(data)  # 获取数据长度
    confidence_interval = 1.96 / np.sqrt(n)  # 计算95%置信区间边界

    # 找出显著的滞后阶数
    significant_acf = [i for i in range(1, len(acf_values)) if abs(acf_values[i]) > confidence_interval]  # 筛选显著的ACF滞后阶数
    significant_pacf = [i for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > confidence_interval]  # 筛选显著的PACF滞后阶数

    print(f"显著的ACF滞后阶数: {significant_acf[:5]}")  # 只显示前5个显著的ACF滞后阶数
    print(f"显著的PACF滞后阶数: {significant_pacf[:5]}")  # 只显示前5个显著的PACF滞后阶数

    # 建议参数
    if len(significant_pacf) > 0:
        suggested_p = min(3, max(significant_pacf[:3]))  # 根据PACF建议AR阶数p，限制在3以内
    else:
        suggested_p = 0

    if len(significant_acf) > 0:
        suggested_q = min(3, max(significant_acf[:3]))  # 根据ACF建议MA阶数q，限制在3以内
    else:
        suggested_q = 0

    print(f"建议的AR阶数 (p): {suggested_p}")  # 打印建议的AR阶数
    print(f"建议的MA阶数 (q): {suggested_q}")  # 打印建议的MA阶数

    return suggested_p, suggested_q  # 返回建议参数


suggested_p, suggested_q = analyze_acf_pacf(analysis_data)  # 调用函数分析ACF和PACF

# 2.通过网格搜索法利用AIC和BIC评分来获取ARIMA的三个参数p,d,q
# 网格搜索最优ARIMA参数
def grid_search_arima(data, max_p=3, max_d=2, max_q=3):
    """
    网格搜索最优ARIMA参数
    遍历所有参数组合，返回最优参数和搜索结果
    """
    import itertools  # 导入迭代工具，用于生成参数组合

    # 生成所有可能的参数组合
    p_values = range(0, max_p + 1)  # AR阶数p的取值范围
    d_values = range(0, max_d + 1)  # 差分阶数d的取值范围
    q_values = range(0, max_q + 1)  # MA阶数q的取值范围

    best_aic = float('inf')  # 初始化最佳AIC值为无穷大
    best_params = None  # 初始化最佳参数组合
    results_list = []  # 存储所有结果的列表

    print("开始网格搜索最优ARIMA参数...")  # 打印开始搜索提示
    print("参数组合评估:")  # 打印评估标题

    for p, d, q in itertools.product(p_values, d_values, q_values):  # 遍历所有参数组合
        try:
            model = ARIMA(data, order=(p, d, q))  # 创建ARIMA模型
            fitted_model = model.fit()  # 拟合模型

            aic = fitted_model.aic  # 获取AIC值
            bic = fitted_model.bic  # 获取BIC值

            results_list.append({  # 将结果添加到列表
                'params': (p, d, q),
                'AIC': aic,
                'BIC': bic
            })

            print(f"ARIMA({p},{d},{q}) - AIC: {aic:.2f}, BIC: {bic:.2f}")  # 打印当前模型评分

            if aic < best_aic:  # 如果当前AIC更小，更新最佳参数
                best_aic = aic
                best_params = (p, d, q)

        except Exception as e:  # 捕获拟合失败的情况
            print(f"ARIMA({p},{d},{q}) - 拟合失败: {str(e)[:50]}")  # 打印错误信息
            continue

    # 转换为DataFrame便于分析
    results_df = pd.DataFrame(results_list)  # 将结果列表转换为DataFrame

    print(f"\n最优参数: ARIMA{best_params}")  # 打印最优参数组合
    print(f"最佳AIC值: {best_aic:.2f}")  # 打印最佳AIC值

    return best_params, results_df  # 返回最优参数和所有结果


# 对收益率序列进行网格搜索（d=0，因为收益率已经平稳）
best_params, search_results = grid_search_arima(stock_data['returns'].dropna(), max_p=3, max_d=1, max_q=3)  # 执行网格搜索

# 显示前10个最佳模型
print("\n前10个最佳模型（按AIC排序）:")  # 打印标题
top_models = search_results.nsmallest(10, 'AIC')  # 按AIC排序取前10个模型
for idx, row in top_models.iterrows():  # 遍历前10个模型
    print(f"ARIMA{row['params']} - AIC: {row['AIC']:.2f}, BIC: {row['BIC']:.2f}")  # 打印模型评分

# 模型拟合和参数估计
# 使用最优参数拟合ARIMA模型
optimal_model = ARIMA(stock_data['returns'].dropna(), order=best_params)  # 使用最优参数创建ARIMA模型
fitted_model = optimal_model.fit()  # 拟合最优模型

# 显示模型摘要
print("最优ARIMA模型摘要:")  # 打印模型摘要标题
print(fitted_model.summary())  # 显示模型详细摘要（系数、标准误、p值等）

# 提取模型参数
params = fitted_model.params  # 获取模型参数估计值
print(f"\n模型参数估计值:")  # 打印参数标题
for param_name, param_value in params.items():  # 遍历所有参数
    print(f"{param_name}: {param_value:.6f}")  # 打印参数名称和估计值

# 模型预测
# 预测未来30天的收益率
forecast_steps = 30  # 设置预测步数为30天
forecast = fitted_model.forecast(steps=forecast_steps)  # 预测未来30天收益率
forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()  # 获取95%置信区间

# 创建预测日期
last_date = stock_data.index[-1]  # 获取最后日期
forecast_dates = pd.date_range(start=last_date + timedelta(days=1),  # 从次日开始
                               periods=forecast_steps, freq='D')  # 生成30个日期，频率为日

# 将收益率预测转换为价格预测
last_price = stock_data['price'].iloc[-1]  # 获取最后价格
predicted_prices = [last_price]  # 初始化预测价格列表

for return_forecast in forecast:  # 遍历收益率预测值
    next_price = predicted_prices[-1] * (1 + return_forecast)  # 根据收益率计算下一期价格
    predicted_prices.append(next_price)  # 添加到列表

predicted_prices = predicted_prices[1:]  # 移除初始价格，保留预测价格

# 计算价格预测的置信区间
price_ci_lower = []  # 初始化价格置信区间下限列表
price_ci_upper = []  # 初始化价格置信区间上限列表
current_price = last_price  # 设置当前价格为最后价格

for i in range(forecast_steps):  # 遍历每个预测步数
    lower_return = forecast_ci.iloc[i, 0]  # 获取收益率置信区间下限
    upper_return = forecast_ci.iloc[i, 1]  # 获取收益率置信区间上限

    # 简化的置信区间计算（实际应该考虑累积效应）
    lower_price = current_price * (1 + lower_return)  # 计算价格下限
    upper_price = current_price * (1 + upper_return)  # 计算价格上限

    price_ci_lower.append(lower_price)  # 添加到下限列表
    price_ci_upper.append(upper_price)  # 添加到上限列表

    current_price = predicted_prices[i]  # 更新当前价格为预测价格

# 预测结果可视化
fig, axes = plt.subplots(2, 1, figsize=(15, 12))  # 创建2行1列的子图布局

# 收益率预测
axes[0].plot(stock_data.index[-60:], stock_data['returns'][-60:],  # 绘制最近60天历史收益率
             label='历史收益率', color='blue', linewidth=1.5)
axes[0].plot(forecast_dates, forecast,  # 绘制预测收益率
             label='预测收益率', color='red', linestyle='--', linewidth=2)
axes[0].fill_between(forecast_dates,  # 绘制置信区间
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1],
                     color='pink', alpha=0.3, label='95%置信区间')
axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)  # 添加零参考线
axes[0].set_title('收益率预测结果', fontsize=14, fontweight='bold')  # 设置标题
axes[0].set_ylabel('收益率')  # 设置Y轴标签
axes[0].legend()  # 显示图例
axes[0].grid(True, alpha=0.3)  # 显示网格线

# 价格预测
axes[1].plot(stock_data.index[-60:], stock_data['price'][-60:],  # 绘制最近60天历史价格
             label='历史价格', color='blue', linewidth=1.5)
axes[1].plot(forecast_dates, predicted_prices,  # 绘制预测价格
             label='预测价格', color='red', linestyle='--', linewidth=2, marker='o')
axes[1].fill_between(forecast_dates, price_ci_lower, price_ci_upper,  # 绘制价格置信区间
                     color='pink', alpha=0.3, label='95%置信区间')
axes[1].set_title('股票价格预测结果', fontsize=14, fontweight='bold')  # 设置标题
axes[1].set_ylabel('价格 (元)')  # 设置Y轴标签
axes[1].set_xlabel('日期')  # 设置X轴标签
axes[1].legend()  # 显示图例
axes[1].grid(True, alpha=0.3)  # 显示网格线

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图形

# 预测结果表格
forecast_df = pd.DataFrame({  # 创建预测结果DataFrame
    '日期': forecast_dates,  # 日期列
    '预测收益率': forecast,  # 预测收益率列
    '收益率下限': forecast_ci.iloc[:, 0],  # 收益率置信区间下限
    '收益率上限': forecast_ci.iloc[:, 1],  # 收益率置信区间上限
    '预测价格': predicted_prices,  # 预测价格列
    '价格下限': price_ci_lower,  # 价格置信区间下限
    '价格上限': price_ci_upper  # 价格置信区间上限
})

print("未来10天预测结果:")  # 打印标题
print(forecast_df.head(10).round(4))  # 显示前10行预测结果，保留4位小数

# 模型评估
# 分割数据进行回测评估
split_ratio = 0.8  # 设置训练集比例为80%
split_point = int(len(stock_data) * split_ratio)  # 计算分割点索引
train_data = stock_data['returns'][:split_point]  # 提取训练集数据
test_data = stock_data['returns'][split_point:]  # 提取测试集数据

print(f"训练集大小: {len(train_data)}")  # 打印训练集大小
print(f"测试集大小: {len(test_data)}")  # 打印测试集大小

# 重新训练模型
train_model = ARIMA(train_data, order=best_params)  # 使用训练数据创建ARIMA模型
train_fitted = train_model.fit()  # 在训练集上拟合模型

# 预测测试期
test_forecast = train_fitted.forecast(steps=len(test_data))  # 预测测试集数据

# 计算评估指标
mae = mean_absolute_error(test_data, test_forecast)  # 计算平均绝对误差（MAE）
rmse = np.sqrt(mean_squared_error(test_data, test_forecast))  # 计算均方根误差（RMSE）
mape = np.mean(np.abs((test_data - test_forecast) / test_data)) * 100  # 计算平均绝对百分比误差（MAPE）

print(f"\n收益率预测精度评估:")  # 打印评估标题
print(f"MAE (平均绝对误差): {mae:.6f}")  # 打印MAE值
print(f"RMSE (均方根误差): {rmse:.6f}")  # 打印RMSE值
print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")  # 打印MAPE值

# 方向准确率（预测涨跌方向的准确率）
actual_direction = np.sign(test_data)  # 计算实际涨跌方向（正数为涨，负数为跌）
predicted_direction = np.sign(test_forecast)  # 计算预测涨跌方向
direction_accuracy = np.mean(actual_direction == predicted_direction)  # 计算方向预测准确率
print(f"方向预测准确率: {direction_accuracy:.2%}")  # 打印方向预测准确率

# 可视化回测结果
plt.figure(figsize=(15, 8))  # 创建新图形，设置大小
plt.plot(test_data.index, test_data, label='实际收益率', color='blue', linewidth=1.5)  # 绘制实际收益率
plt.plot(test_data.index, test_forecast, label='预测收益率', color='red', linestyle='--', linewidth=1.5)  # 绘制预测收益率
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # 添加零参考线
plt.title(f'模型回测结果 - ARIMA{best_params}', fontsize=14, fontweight='bold')  # 设置标题，包含模型参数
plt.xlabel('日期')  # 设置X轴标签
plt.ylabel('收益率')  # 设置Y轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.show()  # 显示图形