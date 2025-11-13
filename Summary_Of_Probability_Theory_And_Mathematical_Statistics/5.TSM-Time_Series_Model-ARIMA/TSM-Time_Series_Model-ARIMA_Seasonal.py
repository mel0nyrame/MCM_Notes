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
import itertools  # 导入迭代工具，用于生成参数组合

# 统计分析库
import statsmodels.api as sm  # 统计分析库，提供统计模型和推断工具
from scipy import stats  # 导入scipy的统计模块，提供概率分布和统计检验函数
from statsmodels.tsa.arima.model import ARIMA  # 导入ARIMA模型，用于时间序列预测和分析
from statsmodels.tsa.seasonal import seasonal_decompose # 导入季节性分解函数，用于季节性分解
from statsmodels.tsa.stattools import adfuller  # 导入ADF检验函数，用于检验时间序列的平稳性
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # 导入绘制自相关和偏自相关图的函数
from statsmodels.stats.diagnostic import acorr_ljungbox  # 导入Ljung-Box检验，用于检测时间序列的自相关性
from statsmodels.tsa.statespace.sarimax import SARIMAX  # 导入SARIMAX模型，用于季节性ARIMA建模

# 忽略警告
import warnings
warnings.filterwarnings('ignore')  # 忽略运行中产生的警告信息，使输出更清晰

# 季节性商品销售预测

# 设置可视化风格
sns.set_style('whitegrid')  # 设置seaborn图表风格为白色网格，提高可读性
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 设置unicode_minus参数为False，用来正常显示负号

# 生成带季节性的销售数据
def generate_seasonal_sales_data(start_date='2022-01-01', periods=24):
    """
    生成具有趋势和季节性的月度销售数据
    模拟电商商品的销售模式：
    - 整体上升趋势
    - 11、12月销售高峰（双十一、双十二）
    - 2、3月销售低谷（春节效应）
    - 随机波动
    """
    np.random.seed(114514)  # 设置随机种子为114514，保证结果可重复

    # 创建月度日期索引
    dates = pd.date_range(start=start_date, periods=periods, freq='M')

    # 基础趋势（缓慢上升）
    trend = 1000 + np.linspace(0, 500, periods) + 50 * np.sin(np.linspace(0, 2 * np.pi, periods))  # 生成基础趋势（线性增长+正弦波动）

    # 季节性成分（12个月周期）
    seasonal_pattern = np.array([
        -150,  # 1月（春节前）
        -200,  # 2月（春节）
        -100,  # 3月（春节后）
        50,  # 4月
        100,  # 5月
        80,  # 6月（618购物节）
        60,  # 7月
        40,  # 8月
        80,  # 9月
        120,  # 10月
        300,  # 11月（双十一）
        250  # 12月（双十二）
    ])

    # 重复季节模式
    seasonal = np.tile(seasonal_pattern, periods // 12 + 1)[:periods]  # 将季节模式重复以覆盖整个数据期

    # 随机噪声
    noise = np.random.normal(0, 80, periods)  # 生成均值为0、标准差为80的正态随机噪声

    # 合成最终销售数据
    sales = trend + seasonal + noise  # 将趋势、季节和噪声成分相加

    # 确保销售量为正数
    sales = np.maximum(sales, 100)  # 将所有小于100的销售值截断为100

    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,  # 日期列
        'sales': sales,  # 销售量列
        'trend': trend,  # 趋势成分列
        'seasonal': seasonal,  # 季节成分列
        'noise': noise  # 噪声成分列
    })
    df.set_index('date', inplace=True)  # 将日期列设置为索引

    return df

# 数据预览
print("\n1. 生成季节性销售数据")  # 打印数据生成标题
print("-" * 30)  # 打印分隔线
sales_data = generate_seasonal_sales_data(periods=24)  # 生成长度为24个月的销售数据
print(f"数据期间: {sales_data.index[0].strftime('%Y-%m')} 至 {sales_data.index[-1].strftime('%Y-%m')}")  # 打印数据时间范围
print(f"数据点数: {len(sales_data)} 个月")  # 打印数据点数
print(f"销售量范围: {sales_data['sales'].min():.0f} - {sales_data['sales'].max():.0f}")  # 打印销售量最小最大值

# 显示基本统计信息
print("\n销售数据基本统计:")  # 打印统计标题
print(sales_data['sales'].describe())  # 显示销售量的描述性统计

# 数据可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 创建2×2的子图布局，设置图形大小

# 销售量时序图
axes[0,0].plot(sales_data.index, sales_data['sales'], marker='o', linewidth=2.5, markersize=6, color='steelblue')  # 绘制带圆形标记的销售量时序图
axes[0,0].set_title('月度销售量时序图', fontsize=14, fontweight='bold')  # 设置子图标题
axes[0,0].set_ylabel('销售量')  # 设置Y轴标签
axes[0,0].grid(True, alpha=0.3)  # 显示网格线
axes[0,0].tick_params(axis='x', rotation=45)  # 旋转X轴标签45度以提高可读性

# 按月份的箱线图（展示季节性）
monthly_sales = sales_data['sales'].groupby(sales_data.index.month)  # 按月份分组计算销售量
month_data = [monthly_sales.get_group(i).values for i in range(1, 13) if i in monthly_sales.groups]  # 提取各月份数据
month_labels = [f'{i}月' for i in range(1, 13) if i in monthly_sales.groups]  # 生成月份标签
axes[0,1].boxplot(month_data, labels=month_labels)  # 绘制箱线图展示各月份分布
axes[0,1].set_title('各月份销售量分布', fontsize=14, fontweight='bold')  # 设置子图标题
axes[0,1].set_ylabel('销售量')  # 设置Y轴标签
axes[0,1].tick_params(axis='x', rotation=45)  # 旋转X轴标签45度

# 年度对比
years = sales_data.index.year.unique()  # 获取数据中的所有年份
for year in years:  # 遍历每个年份
    year_data = sales_data[sales_data.index.year == year]  # 筛选当前年份的数据
    axes[1,0].plot(year_data.index.month, year_data['sales'], marker='o', label=f'{year}年', linewidth=2)  # 绘制月度销售对比线
axes[1,0].set_title('不同年份月度销售对比', fontsize=14, fontweight='bold')  # 设置子图标题
axes[1,0].set_xlabel('月份')  # 设置X轴标签
axes[1,0].set_ylabel('销售量')  # 设置Y轴标签
axes[1,0].legend()  # 显示图例
axes[1,0].grid(True, alpha=0.3)  # 显示网格线
axes[1,0].set_xticks(range(1, 13))  # 设置X轴刻度为1-12月

# 同比增长率
if len(sales_data) >= 12:  # 检查数据是否足够计算同比增长
    yoy_growth = sales_data['sales'].pct_change(12) * 100  # 计算同比增长率（12个月前）
    axes[1,1].plot(yoy_growth.index, yoy_growth, marker='o', color='green', linewidth=2)  # 绘制同比增长率曲线
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)  # 添加零增长参考线
    axes[1,1].set_title('同比增长率', fontsize=14, fontweight='bold')  # 设置子图标题
    axes[1,1].set_xlabel('日期')  # 设置X轴标签
    axes[1,1].set_ylabel('同比增长率 (%)')  # 设置Y轴标签
    axes[1,1].tick_params(axis='x', rotation=45)  # 旋转X轴标签45度
    axes[1,1].grid(True, alpha=0.3)  # 显示网格线

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图形

# 计算月度统计
monthly_stats = sales_data['sales'].groupby(sales_data.index.month).agg(['mean', 'std', 'min', 'max'])  # 按月聚合计算统计量
monthly_stats.index = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'][:len(monthly_stats)]  # 设置月份标签
print("\n各月份销售统计:")  # 打印统计标题
print(monthly_stats.round(1))  # 显示统计结果（保留1位小数）

# 找出销售高峰和低谷
peak_month = monthly_stats['mean'].idxmax()  # 找出平均销售量最高的月份
trough_month = monthly_stats['mean'].idxmin()  # 找出平均销售量最低的月份
print(f"\n销售高峰月份: {peak_month} (平均销量: {monthly_stats.loc[peak_month, 'mean']:.0f})")  # 打印高峰信息
print(f"销售低谷月份: {trough_month} (平均销量: {monthly_stats.loc[trough_month, 'mean']:.0f})")  # 打印低谷信息
print(f"峰谷差异: {monthly_stats.loc[peak_month, 'mean'] - monthly_stats.loc[trough_month, 'mean']:.0f}")  # 打印峰谷差异

# 季节性分解分析
# 执行季节性分解
decomposition = seasonal_decompose(sales_data['sales'], model='additive', period=12)  # 使用加法模型进行季节性分解

# 可视化分解结果
fig, axes = plt.subplots(4, 1, figsize=(16, 14))  # 创建4行1列的子图布局

# 原始序列
decomposition.observed.plot(ax=axes[0], title='原始销售数据', color='blue', linewidth=2)  # 绘制原始序列
axes[0].set_ylabel('销售量')  # 设置Y轴标签
axes[0].grid(True, alpha=0.3)  # 显示网格线

# 趋势成分
decomposition.trend.plot(ax=axes[1], title='趋势成分', color='red', linewidth=2)  # 绘制趋势成分
axes[1].set_ylabel('趋势')  # 设置Y轴标签
axes[1].grid(True, alpha=0.3)  # 显示网格线

# 季节成分
decomposition.seasonal.plot(ax=axes[2], title='季节成分', color='green', linewidth=2)  # 绘制季节成分
axes[2].set_ylabel('季节效应')  # 设置Y轴标签
axes[2].grid(True, alpha=0.3)  # 显示网格线

# 残差成分
decomposition.resid.plot(ax=axes[3], title='残差成分', color='orange', linewidth=1.5)  # 绘制残差成分
axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.7)  # 添加零参考线
axes[3].set_ylabel('残差')  # 设置Y轴标签
axes[3].set_xlabel('日期')  # 设置X轴标签
axes[3].grid(True, alpha=0.3)  # 显示网格线

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图形

# 分析季节性模式
seasonal_pattern = decomposition.seasonal[:12]  # 提取一年的季节模式
print("季节性分析结果:")  # 打印分析标题
print(f"季节性强度 (标准差): {seasonal_pattern.std():.1f}")  # 打印季节性强度
print(f"最强正季节效应: {seasonal_pattern.max():.1f} (第{seasonal_pattern.idxmax().month}月)")  # 打印最强正效应
print(f"最强负季节效应: {seasonal_pattern.min():.1f} (第{seasonal_pattern.idxmin().month}月)")  # 打印最强负效应

# 趋势分析
trend_component = decomposition.trend.dropna()  # 提取趋势成分并去除缺失值
trend_slope = np.polyfit(range(len(trend_component)), trend_component, 1)[0]  # 计算线性趋势斜率
print(f"平均月度增长趋势: {trend_slope:.1f} 单位/月")  # 打印趋势增长率

# 1.对数据进行平稳性检验和差分
def comprehensive_stationarity_test(data, title):
    """
    综合平稳性检验（ADF+KPSS）
    返回布尔值表示是否平稳
    """
    print(f"\n{title}:")  # 打印检验标题

    # ADF检验
    adf_result = adfuller(data.dropna())  # 执行ADF检验并删除缺失值
    print(f"ADF统计量: {adf_result[0]:.4f}")  # 打印ADF统计量
    print(f"p值: {adf_result[1]:.6f}")  # 打印p值

    if adf_result[1] <= 0.05:  # 判断p值是否小于显著性水平0.05
        print("ADF检验结论: 序列平稳")  # ADF检验平稳结论
        is_stationary = True
    else:
        print("ADF检验结论: 序列非平稳")  # ADF检验非平稳结论
        is_stationary = False

    # KPSS检验（互补检验）
    try:
        from statsmodels.tsa.stattools import kpss  # 导入KPSS检验函数
        kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = kpss(data.dropna())  # 执行KPSS检验
        print(f"KPSS统计量: {kpss_stat:.4f}")  # 打印KPSS统计量
        print(f"KPSS p值: {kpss_pvalue:.6f}")  # 打印KPSS p值

        if kpss_pvalue >= 0.05:  # KPSS检验的判断标准与ADF相反
            print("KPSS检验结论: 序列平稳")  # KPSS检验平稳结论
        else:
            print("KPSS检验结论: 序列非平稳")  # KPSS检验非平稳结论
    except:
        print("KPSS检验失败")  # KPSS检验异常处理

    return is_stationary  # 返回平稳性判断结果


# 原始序列检验
original_stationary = comprehensive_stationarity_test(sales_data['sales'], "原始销售序列")  # 检验原始销售序列的平稳性

# 一阶差分检验
sales_diff1 = sales_data['sales'].diff().dropna()  # 计算一阶差分并删除缺失值
diff1_stationary = comprehensive_stationarity_test(sales_diff1, "一阶差分序列")  # 检验一阶差分序列的平稳性

# 季节差分检验
sales_seasonal_diff = sales_data['sales'].diff(12).dropna()  # 计算12期季节差分并删除缺失值
seasonal_diff_stationary = comprehensive_stationarity_test(sales_seasonal_diff, "季节差分序列 (12期)")  # 检验季节差分序列的平稳性

# 一阶+季节差分检验
sales_both_diff = sales_data['sales'].diff().diff(12).dropna()  # 计算一阶+季节差分并删除缺失值
both_diff_stationary = comprehensive_stationarity_test(sales_both_diff, "一阶+季节差分序列")  # 检验组合差分序列的平稳性

# 平稳性检验可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 创建2×2的子图布局

# 原始序列
axes[0, 0].plot(sales_data['sales'], color='blue', linewidth=2)  # 绘制原始销售序列
axes[0, 0].set_title('原始销售序列 (非平稳)', fontsize=12, fontweight='bold')  # 设置子图标题
axes[0, 0].set_ylabel('销售量')  # 设置Y轴标签
axes[0, 0].grid(True, alpha=0.3)  # 显示网格线

# 一阶差分
axes[0, 1].plot(sales_diff1, color='green', linewidth=1.5)  # 绘制一阶差分序列
axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)  # 添加零参考线
axes[0, 1].set_title('一阶差分序列', fontsize=12, fontweight='bold')  # 设置子图标题
axes[0, 1].set_ylabel('差分值')  # 设置Y轴标签
axes[0, 1].grid(True, alpha=0.3)  # 显示网格线

# 季节差分
axes[1, 0].plot(sales_seasonal_diff, color='orange', linewidth=1.5)  # 绘制季节差分序列
axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)  # 添加零参考线
axes[1, 0].set_title('季节差分序列 (12期)', fontsize=12, fontweight='bold')  # 设置子图标题
axes[1, 0].set_ylabel('季节差分值')  # 设置Y轴标签
axes[1, 0].set_xlabel('日期')  # 设置X轴标签
axes[1, 0].grid(True, alpha=0.3)  # 显示网格线

# 一阶+季节差分
axes[1, 1].plot(sales_both_diff, color='purple', linewidth=1.5)  # 绘制一阶+季节差分序列
axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)  # 添加零参考线
axes[1, 1].set_title('一阶+季节差分序列', fontsize=12, fontweight='bold')  # 设置子图标题
axes[1, 1].set_ylabel('组合差分值')  # 设置Y轴标签
axes[1, 1].set_xlabel('日期')  # 设置X轴标签
axes[1, 1].grid(True, alpha=0.3)  # 显示网格线

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图形

# 确定差分阶数
print(f"\n差分阶数确定:")  # 打印差分阶数确定标题
print(f"普通差分次数 (d): 1")  # 打印普通差分次数
print(f"季节差分次数 (D): 1")  # 打印季节差分次数
print(f"季节周期 (s): 12")  # 打印季节周期

# ACF和PACF分析
# 对平稳化后的序列进行ACF/PACF分析
if len(sales_both_diff) > 0:  # 判断组合差分序列是否有数据
    analysis_data = sales_both_diff.dropna()  # 使用组合差分序列
else:
    analysis_data = sales_diff1  # 否则使用一阶差分序列

# 计算合适的lags参数，确保不超过数据长度的一半
max_lags = min(20, len(analysis_data) // 2)  # 设置最大滞后期为20或数据长度的一半

# 绘制ACF和PACF图
fig, axes = plt.subplots(2, 2, figsize=(16, 10))  # 创建2×2的子图布局

# 非季节性ACF和PACF
plot_acf(analysis_data, ax=axes[0, 0], lags=max_lags, title=f'ACF图 (非季节性, lags={max_lags})')  # 绘制非季节性ACF图
axes[0, 0].grid(True, alpha=0.3)  # 显示网格线

plot_pacf(analysis_data, ax=axes[0, 1], lags=max_lags, title=f'PACF图 (非季节性, lags={max_lags})')  # 绘制非季节性PACF图
axes[0, 1].grid(True, alpha=0.3)  # 显示网格线

# 季节性ACF和PACF（如果数据足够长）
if len(analysis_data) > 24:  # 判断数据长度是否足够分析季节性
    seasonal_lags = min(36, len(analysis_data) // 2)  # 设置季节性滞后期为36或数据长度的一半
    plot_acf(analysis_data, ax=axes[1, 0], lags=seasonal_lags, title=f'ACF图 (包含季节性, lags={seasonal_lags})')  # 绘制包含季节性的ACF图
    axes[1, 0].grid(True, alpha=0.3)  # 显示网格线

    plot_pacf(analysis_data, ax=axes[1, 1], lags=seasonal_lags, title=f'PACF图 (包含季节性, lags={seasonal_lags})')  # 绘制包含季节性的PACF图
    axes[1, 1].grid(True, alpha=0.3)  # 显示网格线
else:
    axes[1, 0].set_title('数据长度不足，无法显示季节性ACF图')  # 显示数据不足提示
    axes[1, 1].set_title('数据长度不足，无法显示季节性PACF图')  # 显示数据不足提示

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图形

# 自动参数识别建议
def suggest_sarima_parameters(data, seasonal_period=12):
    """
    基于ACF/PACF特征建议SARIMA参数
    返回非季节性参数(p,d,q)和季节性参数(P,D,Q,s)
    """
    from statsmodels.tsa.stattools import acf, pacf  # 导入计算ACF和PACF的函数

    # 计算ACF和PACF
    acf_vals = acf(data, nlags=min(len(data) // 4, 20), fft=False)  # 计算自相关系数
    pacf_vals = pacf(data, nlags=min(len(data) // 4, 20))  # 计算偏自相关系数

    # 置信区间
    n = len(data)  # 获取数据长度
    ci = 1.96 / np.sqrt(n)  # 计算95%置信区间边界

    # 非季节性参数建议
    significant_pacf = sum(abs(pacf_vals[1:4]) > ci)  # 统计前3阶显著的PACF数量
    significant_acf = sum(abs(acf_vals[1:4]) > ci)  # 统计前3阶显著的ACF数量

    p_suggest = min(significant_pacf, 2)  # 建议AR阶数p（限制在2以内）
    q_suggest = min(significant_acf, 2)  # 建议MA阶数q（限制在2以内）

    # 季节性参数建议（简化版）
    if len(data) > seasonal_period:  # 判断数据长度是否超过季节周期
        seasonal_acf = abs(acf_vals[seasonal_period]) if len(acf_vals) > seasonal_period else 0  # 获取季节性ACF值
        seasonal_pacf = abs(pacf_vals[seasonal_period]) if len(pacf_vals) > seasonal_period else 0  # 获取季节性PACF值

        P_suggest = 1 if seasonal_pacf > ci else 0  # 建议季节性AR阶数P
        Q_suggest = 1 if seasonal_acf > ci else 0  # 建议季节性MA阶数Q
    else:
        P_suggest = Q_suggest = 0  # 数据不足时不建议季节性参数

    return p_suggest, 1, q_suggest, P_suggest, 1, Q_suggest, seasonal_period  # 返回建议参数（固定d=1,D=1）


p, d, q, P, D, Q, s = suggest_sarima_parameters(analysis_data)  # 调用函数获取建议参数
print(f"建议的SARIMA参数:")  # 打印建议参数标题
print(f"非季节性部分 (p,d,q): ({p},{d},{q})")  # 打印非季节性参数
print(f"季节性部分 (P,D,Q,s): ({P},{D},{Q},{s})")  # 打印季节性参数

# 2.通过网格搜索法利用AIC和BIC评分来获取ARIMA的三个参数p,d,q
def sarima_grid_search(data, max_order=2, seasonal_period=12):
    """
    SARIMA模型网格搜索
    遍历所有参数组合并返回最优参数和搜索结果
    """
    # 定义参数范围
    p = d = q = range(0, max_order + 1)  # 非季节性参数范围
    P = D = Q = range(0, 2)  # 季节性参数通常较小（0或1）

    # 生成所有参数组合
    pdq = list(itertools.product(p, [1], q))  # 生成非季节性参数组合（固定d=1）
    seasonal_pdq = [(x[0], x[1], x[2], seasonal_period) for x in itertools.product(P, [1], Q)]  # 生成季节性参数组合（固定D=1）

    best_aic = float('inf')  # 初始化最佳AIC值为无穷大
    best_params = None  # 初始化最优非季节性参数
    best_seasonal_params = None  # 初始化最优季节性参数
    results = []  # 存储所有搜索结果的列表

    print("开始SARIMA参数网格搜索...")  # 打印搜索开始提示
    print("格式: SARIMA(p,d,q)(P,D,Q,s) - AIC值")  # 打印输出格式说明

    for param in pdq:  # 遍历非季节性参数组合
        for param_seasonal in seasonal_pdq:  # 遍历季节性参数组合
            try:
                model = SARIMAX(data, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)  # 创建SARIMAX模型
                fitted_model = model.fit(disp=False)  # 拟合模型（不显示迭代信息）

                aic = fitted_model.aic  # 获取AIC值
                bic = fitted_model.bic  # 获取BIC值

                results.append({  # 将结果添加到列表
                    'params': param,
                    'seasonal_params': param_seasonal,
                    'AIC': aic,
                    'BIC': bic
                })

                param_str = f"SARIMA{param}{param_seasonal}"  # 生成参数字符串
                print(f"{param_str} - AIC: {aic:.2f}")  # 打印当前模型的AIC值

                if aic < best_aic:  # 如果当前AIC更好（更小）
                    best_aic = aic  # 更新最佳AIC值
                    best_params = param  # 更新最优非季节性参数
                    best_seasonal_params = param_seasonal  # 更新最优季节性参数

            except Exception as e:  # 捕获模型拟合失败的情况
                param_str = f"SARIMA{param}{param_seasonal}"  # 生成参数字符串
                print(f"{param_str} - 拟合失败")  # 打印拟合失败提示
                continue

    results_df = pd.DataFrame(results)  # 将结果列表转换为DataFrame

    print(f"\n最优SARIMA参数: {best_params}{best_seasonal_params}")  # 打印最优参数组合
    print(f"最佳AIC值: {best_aic:.2f}")  # 打印最佳AIC值

    return best_params, best_seasonal_params, results_df  # 返回最优参数和搜索结果


# 执行网格搜索
best_params, best_seasonal_params, search_results = sarima_grid_search(sales_data['sales'])  # 对销售数据执行网格搜索

# 显示前10个最佳模型
if not search_results.empty:  # 判断搜索结果是否为空
    print("\n前10个最佳模型（按AIC排序）:")  # 打印标题
    top_models = search_results.nsmallest(10, 'AIC')  # 按AIC升序排列取前10个模型
    for idx, row in top_models.iterrows():  # 遍历前10个模型
        params_str = f"SARIMA{row['params']}{row['seasonal_params']}"  # 生成参数字符串
        print(f"{params_str} - AIC: {row['AIC']:.2f}, BIC: {row['BIC']:.2f}")  # 打印模型评分

# 最优SARIMA模型拟合
# 使用最优参数拟合SARIMA模型
optimal_sarima = SARIMAX(sales_data['sales'], order=best_params, seasonal_order=best_seasonal_params, enforce_stationarity=False, enforce_invertibility=False)  # 使用最优参数创建SARIMAX模型
fitted_sarima = optimal_sarima.fit(disp=False)  # 拟合最优模型（不显示迭代信息）

# 显示模型摘要
print("SARIMA模型拟合结果:")  # 打印模型拟合结果标题
print(f"模型规格: SARIMA{best_params}{best_seasonal_params}")  # 打印模型规格
print(f"AIC: {fitted_sarima.aic:.2f}")  # 打印AIC值
print(f"BIC: {fitted_sarima.bic:.2f}")  # 打印BIC值
print(f"对数似然值: {fitted_sarima.llf:.2f}")  # 打印对数似然值

# 模型参数显著性检验
print("\n模型参数估计:")  # 打印参数估计标题
print(fitted_sarima.summary().tables[1])  # 显示模型参数表（系数、标准误、p值等）

# 模型预测
# 设置预测参数
forecast_periods = 6  # 预测未来6个月
confidence_level = 0.95  # 置信水平为95%

# 进行预测
forecast_result = fitted_sarima.get_forecast(steps=forecast_periods)  # 获取预测结果
forecast_mean = forecast_result.predicted_mean  # 提取预测均值
forecast_ci = forecast_result.conf_int(alpha=1-confidence_level)  # 提取置信区间

# 创建预测日期索引
last_date = sales_data.index[-1]  # 获取最后日期
forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_periods, freq='M')  # 生成未来6个月的日期索引

# 创建预测结果DataFrame
forecast_df = pd.DataFrame({
    'forecast': forecast_mean.values,  # 预测值列
    'lower_ci': forecast_ci.iloc[:, 0].values,  # 置信区间下限列
    'upper_ci': forecast_ci.iloc[:, 1].values  # 置信区间上限列
}, index=forecast_index)  # 设置预测日期为索引

print("预测结果:")  # 打印预测结果标题
print(forecast_df.round(0))  # 显示预测结果（四舍五入到整数）

# 计算预测区间宽度
forecast_df['interval_width'] = forecast_df['upper_ci'] - forecast_df['lower_ci']  # 计算预测区间宽度
print(f"\n平均预测区间宽度: {forecast_df['interval_width'].mean():.0f}")  # 打印平均区间宽度
print(f"预测不确定性系数: {(forecast_df['interval_width'].mean() / forecast_df['forecast'].mean() * 100):.1f}%")  # 打印相对不确定性

# 预测可视化
# 创建预测可视化
fig, axes = plt.subplots(2, 1, figsize=(16, 12))  # 创建2行1列的子图布局

# 整体预测图
# 历史数据
axes[0].plot(sales_data.index, sales_data['sales'], 'o-', linewidth=2.5, markersize=6, color='steelblue', label='历史销售数据')  # 绘制历史销售数据

# 预测数据
axes[0].plot(forecast_index, forecast_df['forecast'], 'o-', linewidth=2.5, markersize=6, color='red', label='预测值')  # 绘制预测值

# 置信区间
axes[0].fill_between(forecast_index, forecast_df['lower_ci'], forecast_df['upper_ci'], alpha=0.3, color='red', label=f'{int(confidence_level*100)}%置信区间')  # 绘制置信区间

axes[0].axvline(x=sales_data.index[-1], color='gray', linestyle='--', alpha=0.7, label='预测起点')  # 添加预测起点标记线
axes[0].set_title('SARIMA销售预测结果', fontsize=14, fontweight='bold')  # 设置子图标题
axes[0].set_ylabel('销售量')  # 设置Y轴标签
axes[0].legend()  # 显示图例
axes[0].grid(True, alpha=0.3)  # 显示网格线
axes[0].tick_params(axis='x', rotation=45)  # 旋转X轴标签45度

# 模型拟合效果图
fitted_values = fitted_sarima.fittedvalues  # 获取模型拟合值
axes[1].plot(sales_data.index, sales_data['sales'], 'o-', linewidth=2, markersize=4, color='blue', label='实际值', alpha=0.7)  # 绘制实际值
axes[1].plot(sales_data.index, fitted_values, 'o-', linewidth=2, markersize=4, color='orange', label='拟合值', alpha=0.8)  # 绘制拟合值

axes[1].set_title('模型拟合效果对比', fontsize=14, fontweight='bold')  # 设置子图标题
axes[1].set_ylabel('销售量')  # 设置Y轴标签
axes[1].set_xlabel('日期')  # 设置X轴标签
axes[1].legend()  # 显示图例
axes[1].grid(True, alpha=0.3)  # 显示网格线
axes[1].tick_params(axis='x', rotation=45)  # 旋转X轴标签45度

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图形

# 预测模型精度评估
# 计算拟合精度指标
mae_fit = mean_absolute_error(sales_data['sales'], fitted_values)  # 计算平均绝对误差（MAE）
mse_fit = mean_squared_error(sales_data['sales'], fitted_values)  # 计算均方误差（MSE）
rmse_fit = np.sqrt(mse_fit)  # 计算均方根误差（RMSE）
mape_fit = np.mean(np.abs((sales_data['sales'] - fitted_values) / sales_data['sales'])) * 100  # 计算平均绝对百分比误差（MAPE）

print("样本内拟合精度:")  # 打印评估标题
print(f"MAE (平均绝对误差): {mae_fit:.2f}")  # 打印MAE
print(f"MSE (均方误差): {mse_fit:.2f}")  # 打印MSE
print(f"RMSE (均方根误差): {rmse_fit:.2f}")  # 打印RMSE
print(f"MAPE (平均绝对百分比误差): {mape_fit:.2f}%")  # 打印MAPE

# 相对精度
relative_rmse = rmse_fit / sales_data['sales'].mean() * 100  # 计算相对RMSE（相对于均值的比例）
print(f"相对RMSE: {relative_rmse:.2f}%")  # 打印相对RMSE

# 如果有足够的历史数据，进行滚动预测验证
if len(sales_data) > 18:  # 检查数据长度是否足够进行滚动验证
    print("\n滚动预测验证 (最后6个月):")  # 打印验证标题

    # 用前面的数据训练，预测后面的数据
    train_size = len(sales_data) - 6  # 计算训练集大小（留最后6个月作为测试）
    train_data = sales_data['sales'][:train_size]  # 提取训练数据
    test_data = sales_data['sales'][train_size:]  # 提取测试数据

    # 重新拟合模型
    validation_model = SARIMAX(train_data, order=best_params, seasonal_order=best_seasonal_params, enforce_stationarity=False, enforce_invertibility=False)  # 使用训练数据创建模型
    validation_fitted = validation_model.fit(disp=False)  # 在训练集上拟合模型

    # 预测测试期
    validation_forecast = validation_fitted.get_forecast(steps=len(test_data))  # 预测测试期数据
    validation_pred = validation_forecast.predicted_mean  # 提取预测值

    # 计算预测精度
    mae_test = mean_absolute_error(test_data, validation_pred)  # 计算测试集MAE
    mse_test = mean_squared_error(test_data, validation_pred)  # 计算测试集MSE
    rmse_test = np.sqrt(mse_test)  # 计算测试集RMSE
    mape_test = np.mean(np.abs((test_data - validation_pred) / test_data)) * 100  # 计算测试集MAPE

    print("样本外预测精度:")  # 打印样本外精度标题
    print(f"MAE: {mae_test:.2f}")  # 打印测试集MAE
    print(f"RMSE: {rmse_test:.2f}")  # 打印测试集RMSE
    print(f"MAPE: {mape_test:.2f}%")  # 打印测试集MAPE

    # 预测vs实际对比
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))  # 创建1个子图
    ax.plot(test_data.index, test_data.values, 'o-', label='实际值', linewidth=2, markersize=6)  # 绘制实际值
    ax.plot(test_data.index, validation_pred.values, 's-', label='预测值', linewidth=2, markersize=6)  # 绘制预测值（方形标记）
    ax.set_title('滚动预测验证结果', fontsize=14, fontweight='bold')  # 设置标题
    ax.set_ylabel('销售量')  # 设置Y轴标签
    ax.legend()  # 显示图例
    ax.grid(True, alpha=0.3)  # 显示网格线
    plt.xticks(rotation=45)  # 旋转X轴标签45度
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图形