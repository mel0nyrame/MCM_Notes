# 数据处理和科学计算库
import numpy as np # 导入numpy库，用于进行科学计算
import pandas as pd # 导入pandas库，用于处理数据表

# 可视化库
import matplotlib.pyplot as plt # 导入matplotlib库，用于可视化数据
import seaborn as sns # 导入seaborn库，相比于matplotlib库有更多的函数，能够处理一些较为复杂的图
from scipy.optimize import curve_fit

# 机器学习库
from sklearn.model_selection import train_test_split # 导入sklearn(机器学习)库，用于分割训练数据和测试数据(一般80%用于训练,20%用于检验模型)
from sklearn.linear_model import LinearRegression # 导入线性回归模型库
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # 导入三个评价指标（MSE.R^2,MAE）
from sklearn.model_selection import KFold # ?

# 统计分析库
import statsmodels.api as sm # 统计分析库
from scipy import stats # 统计分析库

# 咖啡冷却温度预测 - 牛顿冷却定律的非线性回归实现

# 设置可视化风格
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 根据真实物理规律生成的数据
def generate_coffee_data(n_points=20, noise_level=1.0):
    """生成咖啡冷却实验数据"""
    np.random.seed(42)

    # 实际测量的时间点（分钟）
    time_data = np.linspace(0, 60, n_points)

    # 真实的冷却规律：T = T_room + (T_initial - T_room) * exp(-k*t)
    # 参数设定：室温22°C，初始温度95°C，冷却系数0.045
    T_room = 22  # 室温
    T_initial = 95  # 初始温度
    k = 0.045  # 冷却系数

    # 理论温度
    true_temp = T_room + (T_initial - T_room) * np.exp(-k * time_data)

    # 添加测量误差（模拟实际测量的不确定性）
    noise = np.random.normal(0, noise_level, len(time_data))
    measured_temp = true_temp + noise

    return time_data, measured_temp, true_temp, (T_room, T_initial, k)


# 生成数据
time_data, measured_temp, true_temp, true_params = generate_coffee_data(n_points=25, noise_level=1.2)

# 创建数据框
df = pd.DataFrame({
    '时间(分钟)': time_data,
    '测量温度(°C)': measured_temp,
    '真实温度(°C)': true_temp,
    '温度差': measured_temp - true_temp
})

# 数据预览
print("\n📊 咖啡冷却实验数据预览：")
print(df.head(10))
print(f"\n📈 数据统计信息：")
print(f"测量时间范围: {time_data.min():.1f} - {time_data.max():.1f} 分钟")
print(f"温度范围: {measured_temp.min():.1f} - {measured_temp.max():.1f} °C")
print(f"平均测量误差: {np.mean(np.abs(df['温度差'])):.2f} °C")

# 图像生成
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 创建2行2列的子图布局，分辨率为1500x1000，fig返回图形对象，axes返回包含4个子图的二维数组

# 原始数据散点图
axes[0, 0].scatter(time_data, measured_temp, color='red', alpha=0.7, s=60, label='测试数据')  # 在左上角子图绘制散点图，time_data为x轴时间数据，measured_temp为y轴测量温度，color='red'设置点颜色为红色，alpha=0.7设置透明度70%（0为完全透明，1为不透明），s=60设置散点大小为60，label='test data'设置图例标签
axes[0, 0].plot(time_data, true_temp, color='blue', linewidth=2, label='真实值')  # 在同一子图绘制真实温度曲线，color='blue'设置线条颜色为蓝色，linewidth=2设置线宽为2个像素，label='Truth'设置图例标签
axes[0, 0].set_xlabel('时间 (分钟)')  # 设置x轴标签文本为"时间 (分钟)"，字体大小默认
axes[0, 0].set_ylabel('温度 (°C)')  # 设置y轴标签文本为"温度 (°C)"，默认字体
axes[0, 0].set_title('咖啡冷却过程')  # 设置子图标题为"咖啡冷却过程"，默认字体大小
axes[0, 0].legend()  # 显示图例，自动使用scatter和plot中设置的label，位置默认为最佳位置
axes[0, 0].grid(True, alpha=0.3)  # 显示网格线，True启用网格，alpha=0.3设置网格线透明度为30%，使网格不遮挡数据

# 温度随时间变化率
temp_diff = np.diff(measured_temp)  # 计算相邻测量温度的差值，返回数组长度比原数据少1，例如[100, 98, 95]变为[-2, -3]
time_diff = np.diff(time_data)  # 计算相邻时间点的差值，通常得到等间隔的时间间隔
cooling_rate = temp_diff / time_diff  # 计算冷却速率，单位°C/分钟，表示每分钟温度下降的幅度

axes[0, 1].plot(time_data[1:], cooling_rate, 'o-', color='green', markersize=5)  # 在右上角子图绘制冷却速率，time_data[1:]从第2个时间点开始（因为diff少一个元素），'o-'设置线条样式为圆圈标记加实线，color='green'设置颜色为绿色，markersize=5设置圆圈大小为5
axes[0, 1].set_xlabel('时间 (分钟)')  # 设置x轴标签
axes[0, 1].set_ylabel('冷却速率 (°C/分钟)')  # 设置y轴标签，明确显示单位
axes[0, 1].set_title('冷却速率随时间变化')  # 设置子图标题
axes[0, 1].grid(True, alpha=0.3)  # 启用透明度30%的网格线

# 测量误差分布
axes[1, 0].hist(df['温度差'], bins=8, color='orange', alpha=0.7, edgecolor='black')  # 在左下角子图绘制直方图，df['温度差']是数据源，bins=8设置分成8个条形柱，color='orange'设置柱体填充色为橙色，alpha=0.7设置70%透明度，edgecolor='black'设置柱体边框为黑色
axes[1, 0].set_xlabel('测量误差 (°C)')  # 设置x轴标签为"测量误差 (°C)"
axes[1, 0].set_ylabel('频次')  # 设置y轴标签为"频次"，表示每个误差范围内出现的次数
axes[1, 0].set_title('测量误差分布')  # 设置子图标题
axes[1, 0].grid(True, alpha=0.3)  # 启用网格线

# 温度vs时间的对数关系
axes[1, 1].scatter(time_data, np.log(measured_temp - 20), color='purple', alpha=0.7)  # 在右下角子图绘制对数变换后的散点图，np.log(measured_temp - 20)计算温度超出环境温度(20°C)的自然对数值，color='purple'设置紫色，alpha=0.7设置透明度
axes[1, 1].set_xlabel('时间 (分钟)')  # 设置x轴标签
axes[1, 1].set_ylabel('ln(温度 - 20)')  # 设置y轴标签，显示对数变换的数学表达式
axes[1, 1].set_title('对数变换后的关系')  # 设置子图标题
axes[1, 1].grid(True, alpha=0.3)  # 启用网格线

plt.tight_layout()  # 自动调整子图间距和布局，防止标签、标题等重叠，确保图形美观紧凑
plt.show()  # 在屏幕上显示最终生成的图形，阻塞代码执行直到图形窗口关闭（如果在脚本中运行）

# 定义非线性回归模型
def newton_cooling_model(t, T_room, T_diff, k):
    """
    牛顿冷却定律模型
    T = T_room + T_diff * exp(-k * t)

    参数：
    t: 时间
    T_room: 室温
    T_diff: 初始温度差 (T_initial - T_room)
    k: 冷却系数
    """
    return T_room + T_diff * np.exp(-k * t)


def exponential_decay_model(t, a, b, c):
    """
    通用指数衰减模型
    T = a + b * exp(-c * t)
    """
    return a + b * np.exp(-c * t)

# 参数估计

# 方法1：直接使用牛顿冷却定律
print("方法1：牛顿冷却定律拟合")  # 打印方法标题，标记第一种建模思路

# 智能初值设定（重要，影响模型拟合效果）
T_room_init = min(measured_temp)  # 用测量数据中的最低温度作为室温的初始猜测值，基于实际数据特征提高拟合成功率
T_diff_init = max(measured_temp) - T_room_init  # 计算初始温度差，即最高温度与室温估计值的差，为冷却幅度提供初始估计
k_init = 0.05  # 设置冷却系数k的初始猜测值为0.05，这是根据经验选择的合理起点，避免算法陷入局部最优

initial_guess_1 = [T_room_init, T_diff_init, k_init]  # 将三个初始参数组合成列表，curve_fit函数需要这种格式作为p0参数
print(f"初始猜测值: T_room={T_room_init:.1f}, T_diff={T_diff_init:.1f}, k={k_init:.3f}")  # 使用f-string格式化输出初始猜测值，:.1f保留1位小数，:.3f保留3位小数，便于用户验证

try:  # 使用try-except结构捕获拟合过程中可能出现的异常，防止程序崩溃
    # 参数拟合
    popt_1, pcov_1 = curve_fit(newton_cooling_model, time_data, measured_temp,
                               p0=initial_guess_1, maxfev=10000)  # 调用curve_fit进行非线性最小二乘拟合，newton_cooling_model是自定义模型函数，p0传入初始猜测，maxfev=10000设置最大迭代次数为10000次，防止收敛过慢

    # 预测
    y_pred_1 = newton_cooling_model(time_data, *popt_1)  # 使用拟合得到的最佳参数popt_1（列表形式）解包后传入模型函数，计算在所有时间点上的预测值，*popt_1将列表元素作为独立参数传递

    # 计算评估指标
    r2_1 = r2_score(measured_temp, y_pred_1)  # 计算决定系数R²，衡量模型解释数据变异的能力，越接近1说明拟合越好，取值范围(-∞,1]
    rmse_1 = np.sqrt(mean_squared_error(measured_temp, y_pred_1))  # 计算均方根误差RMSE，先算均方误差再开方，单位与原始数据一致(°C)，反映预测值与真实值的平均偏差程度
    mae_1 = mean_absolute_error(measured_temp, y_pred_1)  # 计算平均绝对误差MAE，直接取绝对值的平均，对异常值不敏感，更直观反映平均误差大小

    print(f"拟合参数: T_room={popt_1[0]:.2f}°C, T_diff={popt_1[1]:.2f}°C, k={popt_1[2]:.4f}")  # 输出拟合得到的最终参数值，popt_1[0]是室温，popt_1[1]是初始温差，popt_1[2]是冷却系数
    print(f"评估指标: R²={r2_1:.4f}, RMSE={rmse_1:.2f}°C, MAE={mae_1:.2f}°C")  # 输出模型评估指标，保留不同精度的小数位数，便于比较模型性能

    # 参数标准误差
    param_std_1 = np.sqrt(np.diag(pcov_1))  # 从协方差矩阵pcov_1中提取参数标准误差，np.diag获取对角线元素（各参数的方差），再开方得到标准误差，反映参数估计的不确定性
    print(f"参数标准误差: ±{param_std_1[0]:.2f}, ±{param_std_1[1]:.2f}, ±{param_std_1[2]:.4f}")  # 输出各参数的标准误差，以±形式表示，帮助判断参数统计显著性

except RuntimeError as e:  # 捕获拟合失败时的RuntimeError异常，通常由于迭代次数耗尽或无法收敛引起
    print(f"拟合失败: {e}")  # 打印错误信息，e包含具体的失败原因，便于调试

# 方法2：通用指数衰减模型
print("\n方法2：通用指数衰减模型拟合")  # 打印空行和方法标题，\n实现换行，区分不同方法

initial_guess_2 = [T_room_init, T_diff_init, k_init]  # 复用方法1的初始猜测值，尽管模型形式不同，但参数物理意义相似可共享初值

try:  # 为方法2同样设置异常捕获结构
    popt_2, pcov_2 = curve_fit(exponential_decay_model, time_data, measured_temp,
                               p0=initial_guess_2, maxfev=10000)  # 对通用指数衰减模型进行拟合，函数形式与牛顿冷却定律不同但数学结构相似，参数含义为a,b,c而非具体物理量

    y_pred_2 = exponential_decay_model(time_data, *popt_2)  # 用方法2的拟合参数计算预测值，*popt_2解包传递给指数衰减模型

    r2_2 = r2_score(measured_temp, y_pred_2)  # 计算方法2的决定系数R²，与方法1的r2_1对比可评估哪个模型更适合当前数据
    rmse_2 = np.sqrt(mean_squared_error(measured_temp, y_pred_2))  # 计算方法2的RMSE，量化预测误差
    mae_2 = mean_absolute_error(measured_temp, y_pred_2)  # 计算方法2的MAE，提供另一种误差视角

    print(f"拟合参数: a={popt_2[0]:.2f}, b={popt_2[1]:.2f}, c={popt_2[2]:.4f}")  # 输出方法2的拟合参数，a,b,c为通用指数模型参数，无特定物理意义
    print(f"评估指标: R²={r2_2:.4f}, RMSE={rmse_2:.2f}°C, MAE={mae_2:.2f}°C")  # 输出方法2的评估指标，与方法1横向对比

except RuntimeError as e:  # 同样捕获方法2可能产生的拟合错误
    print(f"拟合失败: {e}")  # 打印方法2的具体错误信息，便于定位问题

# 模型比较与可视化

# 比较真实参数与估计参数
print("真实参数 vs 估计参数:")  # 打印子标题，提示后续输出内容为参数对比，便于结果解读
print(f"T_room: 真实值={true_params[0]:.1f}°C, 估计值={popt_1[0]:.1f}°C")  # 使用f-string格式化输出室温参数对比，true_params[0]是模拟时设定的真实室温，popt_1[0]是牛顿冷却模型的拟合结果，:.1f保留1位小数，直观展示估计准确度
print(f"T_diff: 真实值={true_params[1] - true_params[0]:.1f}°C, 估计值={popt_1[1]:.1f}°C")  # 输出初始温差对比，true_params[1]-true_params[0]计算真实初始温差，popt_1[1]是拟合的温差参数，评估模型对温度梯度的捕捉能力
print(f"k: 真实值={true_params[2]:.4f}, 估计值={popt_1[2]:.4f}")  # 输出冷却系数k的对比，true_params[2]是真实的冷却速率常数，popt_1[2]是拟合值，:.4f保留4位小数精确比较

# 创建详细的模型比较图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 创建2行2列的子图布局，分辨率为1600x1200，提供更大的显示区域以容纳详细信息

# 拟合结果对比
time_plot = np.linspace(0, 60, 200)  # 生成从0到60分钟的平滑时间数组，包含200个点，用于绘制连续的理论曲线，比原始数据点更平滑美观
y_true_plot = newton_cooling_model(time_plot, true_params[0], true_params[1] - true_params[0], true_params[2])  # 使用真实参数计算理论冷却曲线，作为基准参考线，评估拟合模型的偏离程度
y_pred_plot_1 = newton_cooling_model(time_plot, *popt_1)  # 使用拟合得到的参数计算牛顿冷却模型的预测曲线，*popt_1解包三个参数(T_room, T_diff, k)
y_pred_plot_2 = exponential_decay_model(time_plot, *popt_2)  # 使用拟合参数计算指数衰减模型预测曲线，*popt_2解包三个通用参数(a, b, c)

axes[0, 0].scatter(time_data, measured_temp, color='red', alpha=0.7, s=60, label='测量数据', zorder=5)  # 在左上角子图绘制原始测量数据散点图，zorder=5确保散点在最上层不被曲线遮挡
axes[0, 0].plot(time_plot, y_true_plot, color='blue', linewidth=2, label='真实模型', linestyle='--')  # 绘制真实模型曲线作为基准，linestyle='--'设置为虚线，与拟合曲线区分，linewidth=2设置线宽2个像素
axes[0, 0].plot(time_plot, y_pred_plot_1, color='green', linewidth=2, label='牛顿冷却拟合')  # 绘制牛顿冷却模型拟合曲线，绿色表示
axes[0, 0].plot(time_plot, y_pred_plot_2, color='orange', linewidth=2, label='指数衰减拟合')  # 绘制指数衰减模型拟合曲线，橙色表示
axes[0, 0].set_xlabel('时间 (分钟)')  # 设置x轴标签为"时间 (分钟)"
axes[0, 0].set_ylabel('温度 (°C)')  # 设置y轴标签为"温度 (°C)"
axes[0, 0].set_title('模型拟合结果比较')  # 设置子图标题，概括该子图内容
axes[0, 0].legend()  # 显示图例，自动收集所有label标签
axes[0, 0].grid(True, alpha=0.3)  # 显示网格线，透明度30%，辅助观察数值

# 残差分析
residuals_1 = measured_temp - y_pred_1  # 计算牛顿冷却模型的残差（测量值减预测值），反映模型在每个数据点的拟合误差，用于诊断模型系统偏差
residuals_2 = measured_temp - y_pred_2  # 计算指数衰减模型的残差

axes[0, 1].scatter(y_pred_1, residuals_1, color='green', alpha=0.7, s=60, label='牛顿冷却模型')  # 在右上角子图绘制牛顿模型的残差散点图，x轴为预测值，y轴为残差，分析残差是否随机分布
axes[0, 1].scatter(y_pred_2, residuals_2, color='orange', alpha=0.7, s=60, label='指数衰减模型')  # 绘制指数衰减模型残差图
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)  # 添加水平参考线y=0，理想情况下残差点应随机分布在此线上下，linestyle='--'设为虚线
axes[0, 1].set_xlabel('预测值 (°C)')  # 设置x轴标签为"预测值 (°C)"
axes[0, 1].set_ylabel('残差 (°C)')  # 设置y轴标签为"残差 (°C)"，显示误差单位
axes[0, 1].set_title('残差分析')  # 设置子图标题，残差分析用于检验模型假设是否成立
axes[0, 1].legend()  # 显示图例
axes[0, 1].grid(True, alpha=0.3)  # 启用网格线

# 误差分布
axes[1, 0].hist(residuals_1, bins=8, alpha=0.7, color='green', label='牛顿冷却模型', edgecolor='black')  # 在左下角子图绘制牛顿模型残差的直方图，bins=8分成8个柱，分析误差分布形态是否接近正态分布
axes[1, 0].hist(residuals_2, bins=8, alpha=0.7, color='orange', label='指数衰减模型', edgecolor='black')  # 绘制指数模型残差直方图，两个hist叠加显示便于比较分布形态差异
axes[1, 0].set_xlabel('残差 (°C)')  # 设置x轴标签
axes[1, 0].set_ylabel('频次')  # 设置y轴标签，表示残差落在每个区间的数据点数量
axes[1, 0].set_title('残差分布')  # 设置子图标题
axes[1, 0].legend()  # 显示图例
axes[1, 0].grid(True, alpha=0.3)  # 启用网格线

# 预测vs实际
axes[1, 1].scatter(measured_temp, y_pred_1, color='green', alpha=0.7, s=60, label='牛顿冷却模型')  # 在右下角子图绘制牛顿模型的预测值vs实际值散点图，若拟合完美则点应落在45度线上
axes[1, 1].scatter(measured_temp, y_pred_2, color='orange', alpha=0.7, s=60, label='指数衰减模型')  # 绘制指数模型的预测vs实际散点
axes[1, 1].plot([measured_temp.min(), measured_temp.max()],
                [measured_temp.min(), measured_temp.max()],
                'r--', linewidth=2, label='理想拟合线')  # 绘制红色虚线表示理想拟合线（45度线），若预测值等于实际值则所有点应落在此线上，[min,max]确定线的起点和终点坐标
axes[1, 1].set_xlabel('实际温度 (°C)')  # 设置x轴标签
axes[1, 1].set_ylabel('预测温度 (°C)')  # 设置y轴标签
axes[1, 1].set_title('预测值 vs 实际值')  # 设置子图标题
axes[1, 1].legend()  # 显示图例
axes[1, 1].grid(True, alpha=0.3)  # 启用网格线

plt.tight_layout()  # 自动调整子图布局，确保标题、轴标签和刻度不重叠，使图形更紧凑美观
plt.show()  # 显示最终生成的模型比较图，阻塞执行直到关闭图形窗口

# 模型诊断

# Shapiro-Wilk正态性检验
stat_1, p_value_1 = stats.shapiro(
    residuals_1)  # 对牛顿冷却模型的残差进行Shapiro-Wilk正态性检验，stats.shapiro()返回两个值：检验统计量stat_1和p值p_value_1，该检验是小样本情况下检验正态性的有效方法，原假设为残差服从正态分布
stat_2, p_value_2 = stats.shapiro(residuals_2)  # 对指数衰减模型的残差进行同样的正态性检验，得到该模型的检验统计量和p值，便于横向比较两个模型残差的正态性优劣

print("残差正态性检验 (Shapiro-Wilk):")  # 打印检验标题，明确说明后续输出的是关于残差正态性的统计检验结果，增强输出可读性
print(
    f"牛顿冷却模型: 统计量={stat_1:.4f}, p值={p_value_1:.4f}")  # 格式化输出牛顿模型的检验结果，stat_1是W统计量(取值范围0-1，越接近1表示越接近正态分布)，p_value_1是显著性概率，:.4f保留4位小数便于精确比较，当p>0.05时认为残差服从正态分布
print(
    f"指数衰减模型: 统计量={stat_2:.4f}, p值={p_value_2:.4f}")  # 输出指数模型的检验结果，通过对比两个模型的p值和统计量可以判断哪个模型的残差更符合正态性假设，这是回归诊断的重要步骤
print(
    "注: p值>0.05表示残差服从正态分布")  # 添加结果解释说明，这是假设检验的判定标准：当p>0.05时不能拒绝原假设(残差正态)，p≤0.05则拒绝原假设，提示残差可能非正态，违背线性回归的基本假设


# 残差自相关性检验
def durbin_watson_test(residuals):  # 定义一个计算Durbin-Watson统计量的自定义函数，用于检测残差序列是否存在自相关性，这是时间序列回归诊断的重要工具，特别适用于按时间顺序采集的数据
    """Durbin-Watson检验统计量"""  # 函数文档字符串，说明函数用途是计算DW检验统计量，帮助用户理解函数功能和使用方法
    n = len(residuals)  # 获取残差序列的长度(样本量n)，DW统计量的计算需要知道样本大小，用于后续统计推断
    if n < 2:  # 设置安全判断条件，当样本量小于2时无法计算DW统计量(至少需要两个点才能计算差分)，防止程序因数据不足而报错
        return None  # 样本不足时返回None，避免程序抛出异常，提高代码健壮性和容错能力

    diff_residuals = np.diff(residuals)  # 计算残差的一阶差分序列，即residuals[i+1] - residuals[i]，DW统计量基于相邻残差的差异构建，用于捕捉残差的序列相关性
    sum_squared_diff = np.sum(diff_residuals ** 2)  # 计算残差差分的平方和，这是DW统计量的分子部分，反映相邻残差变化的累积幅度，值越大说明残差波动越剧烈
    sum_squared_residuals = np.sum(residuals ** 2)  # 计算残差本身的平方和，这是DW统计量的分母部分，反映残差整体的离散程度，作为标准化的基准

    dw = sum_squared_diff / sum_squared_residuals  # 计算DW统计量，公式为Σ(Δe²)/Σ(e²)，其中Δe是残差差分，e是残差，该统计量取值范围在0到4之间，2为理想值
    return dw  # 返回计算得到的DW统计量值，供调用者进行自相关性判断


dw_1 = durbin_watson_test(residuals_1)  # 调用自定义函数计算牛顿冷却模型残差的DW统计量，检测该模型残差是否存在自相关，自相关会导致参数估计无效
dw_2 = durbin_watson_test(residuals_2)  # 计算指数衰减模型残差的DW统计量，便于对比两个模型的残差独立性，理想模型的残差应无自相关

print(f"\n残差自相关性检验 (Durbin-Watson):")  # 打印空行和检验标题，\n实现换行，清晰分隔不同检验部分，提升输出结构清晰度
print(f"牛顿冷却模型: DW={dw_1:.4f}")  # 输出牛顿模型的DW统计量，:.4f保留4位小数，DW值接近2表示无自相关，接近0或4表示存在强正或负自相关，值偏离2越远问题越严重
print(f"指数衰减模型: DW={dw_2:.4f}")  # 输出指数模型的DW统计量，通过对比两个值可以判断哪个模型的残差序列更独立，更符合回归假设
print(
    "注: DW≈2表示无自相关，0<DW<2表示正自相关，2<DW<4表示负自相关")  # 添加解释说明，DW检验的判定规则：DW=2理想(无自相关)，0-2为正相关(残差呈现连续高/低值)，2-4为负相关(残差呈现交替波动)，帮助用户理解检验结果的物理意义

# 交叉验证

def cross_validate_nonlinear(X, y, model_func, initial_guess, cv=5):
    """非线性模型的交叉验证"""  # 定义函数文档字符串，说明该函数用于对非线性模型执行交叉验证，评估模型泛化能力而非仅拟合优度
    kfold = KFold(n_splits=cv, shuffle=True,
                  random_state=42)  # 创建K折交叉验证对象，n_splits=5将数据分为5份，shuffle=True在分割前打乱数据避免顺序影响，random_state=42设置随机种子保证结果可重复

    r2_scores = []  # 初始化空列表，用于存储每次验证得到的R²分数，最终可计算平均R²及其标准差
    rmse_scores = []  # 初始化空列表，存储每次验证的RMSE值，便于后续评估模型预测稳定性

    for train_idx, test_idx in kfold.split(X):  # 遍历KFold生成的训练集和测试集索引，每次迭代使用4/5数据训练，1/5数据验证，共进行5次
        X_train, X_test = X[train_idx], X[test_idx]  # 根据索引分离特征数据，X_train作为训练输入，X_test作为验证输入
        y_train, y_test = y[train_idx], y[test_idx]  # 根据索引分离目标变量，y_train作为训练标签，y_test作为真实值用于评估

        try:  # 设置异常捕获结构，因为非线性拟合可能失败，防止单次失败导致整个交叉验证中断
            # 拟合模型
            popt, _ = curve_fit(model_func, X_train, y_train, p0=initial_guess,
                                maxfev=10000)  # 在训练集上拟合非线性模型，p0传入初始参数，maxfev限制迭代次数，popt返回最优参数，_忽略协方差矩阵

            # 预测
            y_pred = model_func(X_test, *popt)  # 使用拟合参数在测试集上进行预测，*popt解包参数传递给模型函数，得到预测值向量

            # 计算指标
            r2 = r2_score(y_test, y_pred)  # 计算当前折的R²分数，衡量模型在未见数据上的解释能力
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # 计算当前折的RMSE，量化预测误差大小，单位与原始数据一致

            r2_scores.append(r2)  # 将当前R²分数添加到列表，循环结束后计算统计量
            rmse_scores.append(rmse)  # 将当前RMSE添加到列表，用于评估模型稳定性

        except RuntimeError:  # 捕获curve_fit可能失败的异常情况（如无法收敛）
            continue  # 如果当前折拟合失败，跳过该折继续下一折，提高代码鲁棒性

    return np.array(r2_scores), np.array(rmse_scores)  # 将列表转换为NumPy数组返回，便于后续计算均值和标准差等统计量


# 对牛顿冷却模型进行交叉验证
r2_cv_1, rmse_cv_1 = cross_validate_nonlinear(time_data, measured_temp,
                                              newton_cooling_model,
                                              initial_guess_1)  # 调用自定义交叉验证函数评估牛顿冷却模型，传入时间数据、温度数据、模型函数和初始猜测，返回5折的R²和RMSE数组

print(f"牛顿冷却模型交叉验证结果:")  # 打印交叉验证标题，提示后续结果反映模型的泛化性能而非单纯拟合效果
print(f"R²: {r2_cv_1.mean():.4f} ± {r2_cv_1.std():.4f}")  # 输出R²的平均值和标准差，:.4f保留4位小数，平均值反映整体预测能力，标准差反映模型稳定性（波动越小越好）
print(f"RMSE: {rmse_cv_1.mean():.2f} ± {rmse_cv_1.std():.2f} °C")  # 输出RMSE的平均值和标准差，单位°C，平均值反映典型预测误差，标准差反映误差一致性

# 8. 预测未来温度
print("\n🔮 未来温度预测")  # 打印预测章节标题，🔮表情符号增强可视化效果，\n换行分隔章节
print("-" * 40)  # 打印40个短横线作为章节分隔线，提升输出可读性

# 预测未来30分钟的温度
future_time = np.linspace(60, 90, 13)  # 生成未来时间数组，从60到90分钟，包含13个等间距点（每2.5分钟一个），用于外推预测，linspace确保端点包含在内
future_temp_pred = newton_cooling_model(future_time, *popt_1)  # 使用拟合的牛顿冷却模型参数popt_1预测未来时刻的温度，*popt_1解包室温、温差和冷却系数三个参数

# 计算预测区间（基于参数不确定性）
param_std = np.sqrt(np.diag(pcov_1))  # 从参数协方差矩阵提取参数标准误差，np.diag获取对角线元素（各参数估计方差），开方后得到标准误差，反映参数估计的不确定性程度
n_simulations = 1000  # 设置蒙特卡洛模拟次数为1000次，足够多的模拟次数可保证预测区间估计的准确性和稳定性

# 蒙特卡洛模拟预测区间
future_predictions = []  # 初始化空列表，存储每次模拟得到的未来温度预测序列
for _ in range(n_simulations):  # 循环执行1000次蒙特卡洛模拟，每次使用略有不同的参数值来量化预测不确定性
    # 从参数的正态分布中采样
    sampled_params = np.random.normal(popt_1,
                                      param_std)  # 从多元正态分布中随机抽样参数，均值=popt_1(最优参数)，标准差=param_std(参数不确定性)，模拟参数估计的抽样分布
    pred = newton_cooling_model(future_time, *sampled_params)  # 使用抽样得到的参数计算未来温度预测值，*sampled_params解包三个抽样参数
    future_predictions.append(pred)  # 将单次模拟结果添加到列表，循环结束后得到1000条预测轨迹

future_predictions = np.array(future_predictions)  # 将列表转换为三维NumPy数组（1000次模拟×13个时间点），便于后续计算分位数
pred_lower = np.percentile(future_predictions, 2.5, axis=0)  # 计算95%置信区间的下界（第2.5百分位数），axis=0表示对每个时间点跨模拟计算，反映预测值的下限范围
pred_upper = np.percentile(future_predictions, 97.5, axis=0)  # 计算95%置信区间的上界（第97.5百分位数），与下界共同构成95%预测区间，量化预测不确定性

# 创建预测结果表
pred_df = pd.DataFrame({  # 创建Pandas DataFrame存储预测结果，便于表格化展示和后续分析
    '时间(分钟)': future_time,  # 第一列为未来时间点，单位分钟
    '预测温度(°C)': future_temp_pred,  # 第二列为点预测值（使用最优参数），表示最可能的温度
    '95%置信区间下界': pred_lower,  # 第三列为置信区间下限，表示有97.5%概率温度高于此值
    '95%置信区间上界': pred_upper,  # 第四列为置信区间上限，表示有97.5%概率温度低于此值
    '预测区间宽度': pred_upper - pred_lower  # 第五列为区间宽度，量化预测不确定性程度，宽度越大说明预测越不可靠
})  # DataFrame结构便于打印、保存和进一步分析预测结果

print("未来温度预测结果:")  # 打印表格标题
print(pred_df)  # 输出预测结果表格，显示未来每个时间点的温度预测及置信区间，便于用户直观理解预测结果和不确定性

# 可视化预测结果
plt.figure(figsize=(14, 8))  # 创建新图形，figsize=(14,8)设置宽14英寸高8英寸，为预测图提供充足显示空间

# 历史数据
plt.scatter(time_data, measured_temp, color='red', alpha=0.7, s=60, label='历史测量数据',
            zorder=5)  # 绘制历史测量数据散点图，红色表示，zorder=5确保在最上层
plt.plot(time_plot, y_pred_plot_1, color='blue', linewidth=2, label='拟合曲线')  # 绘制历史时段的拟合曲线，蓝色表示

# 预测数据
plt.scatter(future_time, future_temp_pred, color='green', alpha=0.8, s=80,
            marker='s', label='未来预测', zorder=5)  # 绘制未来预测点，绿色方形标记(marker='s')，s=80设置较大尺寸突出显示预测点
plt.fill_between(future_time, pred_lower, pred_upper, alpha=0.3, color='green',
                 label='95%置信区间')  # 填充预测区间，fill_between在上下界之间绘制半透明绿色区域，alpha=0.3设置透明度，直观展示预测不确定性范围

# 连接线
plt.plot(future_time, future_temp_pred, color='green', linewidth=2,
         linestyle='--')  # 用绿色虚线连接预测点，linestyle='--'设置为虚线，与历史实线区分，形成视觉连贯性

plt.axvline(x=60, color='gray', linestyle=':', alpha=0.7,
            label='预测起点')  # 在x=60分钟处添加灰色垂直参考线，标记历史与未来的分界点，linestyle=':'设置为点线
plt.xlabel('时间 (分钟)')  # 设置x轴标签
plt.ylabel('温度 (°C)')  # 设置y轴标签
plt.title('咖啡冷却温度预测（含置信区间）')  # 设置图形标题，明确说明包含置信区间
plt.legend()  # 显示图例，包含所有数据系列的标签
plt.grid(True, alpha=0.3)  # 启用网格线，30%透明度
plt.show()  # 显示预测可视化图形

# 9. 物理意义解释
print("\n🔬 物理意义解释")  # 打印物理意义章节标题，🔬表情符号强调科学性，\n换行分隔
print("-" * 40)  # 打印章节分隔线

estimated_T_room = popt_1[0]  # 从拟合参数提取估计的室温，popt_1[0]是牛顿冷却模型的第一个参数（T_room），表示环境温度
estimated_T_initial = popt_1[0] + popt_1[1]  # 计算估计的初始温度，初始温度=室温+初始温差，popt_1[1]是初始温差参数
estimated_k = popt_1[2]  # 提取估计的冷却系数k，单位是min⁻¹，反映冷却速率快慢，值越大冷却越快

print(f"估计的室温: {estimated_T_room:.1f}°C")  # 输出估计的室温，保留1位小数
print(f"估计的初始温度: {estimated_T_initial:.1f}°C")  # 输出估计的初始温度，反映咖啡初始时刻的温度
print(f"估计的冷却系数: {estimated_k:.4f} min⁻¹")  # 输出冷却系数，保留4位小数，min⁻¹表示每分钟

# 计算半衰期（温度差减半所需时间）
half_life = np.log(2) / estimated_k  # 计算冷却半衰期，公式t₁/₂=ln(2)/k，表示初始温差减少到一半所需时间，ln(2)≈0.693，是指数衰减速率的特征参数
print(f"冷却半衰期: {half_life:.1f}分钟")  # 输出半衰期，保留1位小数，表示咖啡降温速度的时间尺度


# 计算到达特定温度的时间
def time_to_reach_temp(target_temp, T_room, T_initial, k):  # 定义函数计算从初始温度降到目标温度所需时间，基于牛顿冷却定律的逆运算，用于实际场景的时间预估
    """计算到达目标温度所需的时间"""  # 函数文档字符串，说明功能和用途
    if target_temp <= T_room or target_temp >= T_initial:  # 设置输入有效性检查，目标温度必须介于室温和初始温度之间，否则无解或计算无意义
        return None  # 条件不满足时返回None，防止数学计算错误或物理逻辑不合理

    return -np.log((target_temp - T_room) / (
                T_initial - T_room)) / k  # 使用牛顿冷却定律反推时间，公式t = -ln[(T-T_room)/(T_initial-T_room)]/k，通过对数变换求解时间变量


# 计算降到室温+5°C所需的时间
time_to_warm = time_to_reach_temp(estimated_T_room + 5, estimated_T_room, estimated_T_initial,
                                  estimated_k)  # 计算咖啡降到室温以上5°C所需时间，这是常见的适饮温度阈值，estimated_T_room+5表示适饮温度界限
print(f"降到{estimated_T_room + 5:.1f}°C需要: {time_to_warm:.1f}分钟")  # 输出达到适饮温度所需时间，为实际饮用提供时间参考，保留1位小数

# 10. 总结报告
print("\n📋 模型总结报告")  # 打印总结报告标题，📋表情符号表示总结文档，=分隔线强调重要性
print("=" * 60)  # 打印60个等号作为顶级分隔线，突出报告的重要性和正式性

print("🎯 模型性能:")  # 打印性能指标子标题，🎯表情符号强调目标性
print(f"   • 拟合优度 (R²): {r2_1:.4f}")  # 输出决定系数R²，保留4位小数，反映模型对历史数据拟合优度，越接近1越好
print(f"   • 均方根误差 (RMSE): {rmse_1:.2f}°C")  # 输出均方根误差，保留2位小数，量化典型预测误差大小
print(f"   • 平均绝对误差 (MAE): {mae_1:.2f}°C")  # 输出平均绝对误差，保留2位小数，反映平均预测偏差
print(f"   • 交叉验证 R²: {r2_cv_1.mean():.4f} ± {r2_cv_1.std():.4f}")  # 输出交叉验证的R²均值和标准差，±表示波动范围，评估模型泛化能力和稳定性

print("\n📊 参数估计:")  # 打印参数估计子标题，\n换行，📊表情符号表示数据
print(f"   • 室温: {estimated_T_room:.1f} ± {param_std_1[0]:.1f}°C")  # 输出室温估计值及其标准误差，±反映参数估计的精确度
print(f"   • 初始温度: {estimated_T_initial:.1f} ± {param_std_1[1]:.1f}°C")  # 输出初始温度及误差，param_std_1[1]是温差参数的标准误差
print(f"   • 冷却系数: {estimated_k:.4f} ± {param_std_1[2]:.4f} min⁻¹")  # 输出冷却系数及误差，param_std_1[2]是k参数的标准误差，min⁻¹表示单位

print("\n🔍 模型诊断:")  # 打印模型诊断子标题，🔍表情符号表示检查
print(
    f"   • 残差正态性: {'✓' if p_value_1 > 0.05 else '✗'} (p={p_value_1:.4f})")  # 使用三元运算符判断正态性，p>0.05显示✓表示通过检验，否则显示✗，直观显示残差是否符合正态假设
print(
    f"   • 自相关性: {'✓' if abs(dw_1 - 2) < 0.5 else '✗'} (DW={dw_1:.4f})")  # 判断自相关性，DW统计量与2的绝对差<0.5显示✓，表示无显著自相关，否则显示✗

print("\n🌟 实际应用:")  # 打印实际应用子标题，🌟表情符号表示实用性
print(f"   • 冷却半衰期: {half_life:.1f}分钟")  # 输出半衰期，评估冷却速度的时间尺度
print(f"   • 适饮温度时间: {time_to_warm:.1f}分钟")  # 输出达到适饮温度所需时间，提供实用建议
print(f"   • 模型适用范围: 0-90分钟")  # 说明模型推荐的应用时间范围，提醒用户外推风险

print("\n💡 建议:")  # 打印建议子标题，💡表情符号表示启示
if r2_1 > 0.95:  # 判断R²是否大于0.95作为优秀标准
    print("   • 模型拟合优秀，可用于实际预测")  # R²高时给出积极建议
else:
    print("   • 模型拟合良好，但需要更多数据验证")  # R²一般时建议谨慎使用

if p_value_1 > 0.05:  # 判断正态性检验是否通过
    print("   • 残差服从正态分布，模型假设合理")  # 通过时肯定模型假设
else:
    print("   • 残差不服从正态分布，考虑模型改进")  # 不通过时建议改进模型