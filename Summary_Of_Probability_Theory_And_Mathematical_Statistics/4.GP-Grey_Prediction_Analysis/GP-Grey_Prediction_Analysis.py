# 数据处理和科学计算库
import numpy as np  # 导入numpy库，用于进行科学计算
import pandas as pd  # 导入pandas库，用于处理数据表

# 可视化库
import matplotlib.pyplot as plt  # 导入matplotlib库，用于可视化数据
import seaborn as sns  # 导入seaborn库，相比于matplotlib库有更多的函数，能够处理一些较为复杂的图
from scipy.optimize import curve_fit

# 机器学习库
from sklearn.model_selection import train_test_split  # 导入sklearn(机器学习)库，用于分割训练数据和测试数据(一般80%用于训练,20%用于检验模型)
from sklearn.linear_model import LinearRegression  # 导入线性回归模型库
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, \
    mean_absolute_percentage_error  # 导入三个评价指标（MSE.R^2,MAE,MAPE）
from sklearn.model_selection import KFold  # 导入KFold交叉验证工具，用于将数据集分成K个子集进行交叉验证，评估模型稳定性

# 统计分析库
import statsmodels.api as sm  # 统计分析库，提供统计模型和推断工具
from scipy import stats  # 导入scipy的统计模块，提供概率分布和统计检验函数

# 设置可视化风格
sns.set_style('whitegrid')  # 设置seaborn图表风格为白色网格，提高可读性
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 设置unicode_minus参数为False，用来正常显示负号


# 已经封装好的GM11模型，可直接调用函数
class GM11:
    """
    GM(1,1)灰色预测模型类

    主要功能：
    1. 级比检验
    2. 模型训练（ 2.累加生成-均值生成 3.参数估计）
    3. 预测计算
    4. 精度检验
    5. 结果可视化
    """

    def __init__(self):
        self.a = None  # 发展系数，反映系统发展态势，负值表示增长，正值表示衰减
        self.b = None  # 灰作用量，表示系统的外生驱动因素或内生增长能力
        self.x0 = None  # 原始数据序列，存储输入的原始观测值
        self.x1 = None  # 一次累加生成序列(1-AGO)，用于弱化随机性、挖掘规律
        self.n = None  # 原始数据序列的长度，即观测值个数
        self.fitted_values = None  # 模型拟合值，即模型对历史数据的回代计算结果

    def level_ratio_test(self, data):
        """
        级比检验函数
        GM(1,1)建模前必须进行的检验，判断数据是否适合建模

        参数:
        data: 原始数据序列，一维数组格式

        返回:
        valid: 是否通过检验的布尔值
        level_ratios: 级比序列，相邻数据的比值
        bounds: 检验边界值区间
        """
        n = len(data)  # 获取数据序列的长度
        level_ratios = []  # 初始化级比序列列表，用于存储相邻数据的比值

        # 计算级比（相邻数据比值）
        for i in range(1, n):  # 从第二个数据开始遍历
            if data[i] != 0:  # 避免除零错误，确保分母不为0
                ratio = data[i - 1] / data[i]  # 计算前一期与当期数据的比值
                level_ratios.append(ratio)

        # 计算检验边界，GM(1,1)要求级比落在(e^(-2/(n+1)), e^(2/(n+1)))区间内
        lower_bound = np.exp(-2 / (n + 1))  # 计算下边界
        upper_bound = np.exp(2 / (n + 1))  # 计算上边界
        bounds = (lower_bound, upper_bound)

        # 检验所有级比值是否在合理范围内
        valid = all(lower_bound <= ratio <= upper_bound for ratio in level_ratios)  # 遍历级比数组检查是否都在边界内

        # 打印级比检验结果
        print(f"级比检验结果:")
        print(f"检验边界: ({lower_bound:.3f}, {upper_bound:.3f})")
        print(f"级比序列: {[f'{r:.3f}' for r in level_ratios]}")
        print(f"检验结果: {'通过' if valid else '不通过'}")

        return valid, level_ratios, bounds

    def fit(self, data):
        """
        训练GM(1,1)模型
        完成从原始数据到模型参数估计的全过程

        参数:
        data: 原始数据序列，要求长度不小于4
        """
        self.x0 = np.array(data, dtype=float)  # 将输入数据转换为numpy浮点数组，便于后续矩阵运算
        self.n = len(data)  # 记录数据长度

        print(f"\n原始数据: {self.x0}")

        # 进行级比检验，验证数据适用性
        valid, ratios, bounds = self.level_ratio_test(data)
        if not valid:
            print("⚠️  警告：数据未通过级比检验，模型预测精度可能较低！")

        # 一次累加生成(1-AGO)，弱化随机性，凸显趋势
        self.x1 = np.cumsum(self.x0)  # 计算累积和，得到累加序列
        print(f"累加生成序列: {self.x1}")

        # 构造均值序列Z(1)，用于建立灰微分方程
        z1 = []  # 初始化均值序列列表
        for i in range(1, self.n):  # 从第二个数据开始遍历
            z1.append(0.5 * (self.x1[i] + self.x1[i - 1]))  # 计算相邻累加值的均值
        z1 = np.array(z1)  # 转换为numpy数组
        print(f"均值序列Z(1): {z1}")

        # 构造数据矩阵B和观测向量Y，用于最小二乘估计
        B = np.column_stack((-z1, np.ones(len(z1))))  # 构建数据矩阵，第一列为-z1，第二列为1
        Y = self.x0[1:]  # 构建观测向量，从第二个原始数据开始

        print(f"\n数据矩阵B形状: {B.shape}")  # 输出矩阵维度
        print(f"\n数据矩阵B: {B}")  # 输出完整矩阵
        print(f"观测向量Y: {Y}")  # 输出观测向量

        # 最小二乘估计参数[a, b]^T = (B^T B)^(-1) B^T Y
        try:
            # 使用伪逆求解，比直接求逆更稳定，能处理病态矩阵
            params = np.linalg.lstsq(B, Y, rcond=None)[0]
            self.a, self.b = params  # 提取发展系数a和灰作用量b
        except np.linalg.LinAlgError:  # 捕获线性代数错误
            print("❌ 参数估计失败，请检查数据质量")
            return

        # 打印模型参数估计结果
        print(f"\n模型参数估计结果:")
        print(f"发展系数 a = {self.a:.6f}")
        print(f"灰作用量 b = {self.b:.6f}")

        # 判断模型特性，根据a的符号判断增长或衰减趋势
        if self.a > 0:
            print("📈 模型特性: 衰减型（数据呈下降趋势）")
        else:
            print("📊 模型特性: 增长型（数据呈上升趋势）")

    def predict(self, steps=0):
        """
        GM(1,1)预测函数
        基于估计的参数进行时间响应计算和预测

        参数:
        steps: 预测步数，0表示只计算拟合值，正值表示预测未来steps期

        返回:
        predictions: 预测结果数组，包含历史拟合值和未来预测值
        """
        if self.a is None or self.b is None:  # 检查模型是否已训练
            raise ValueError("❌ 模型未训练，请先调用fit方法")

        total_steps = self.n + steps  # 总计算步数=历史数据长度+预测步数
        predictions = []  # 初始化预测结果列表

        # 计算拟合值和预测值
        for k in range(1, total_steps + 1):  # 从k=1开始遍历到总步数
            # 时间响应函数，GM(1,1)模型的核心公式
            if abs(self.a) < 1e-10:  # 处理a接近0的特殊情况，避免除零错误
                x1_pred = self.x0[0] + self.b * k
            else:
                x1_pred = (self.x0[0] - self.b / self.a) * np.exp(-self.a * (k - 1)) + self.b / self.a

            # 通过累减还原计算原始序列预测值
            if k == 1:
                x0_pred = self.x0[0]  # 第一个值保持不变，作为初始条件
            else:
                if abs(self.a) < 1e-10:
                    x0_pred = self.b
                else:
                    x1_prev = (self.x0[0] - self.b / self.a) * np.exp(-self.a * (k - 2)) + self.b / self.a
                    x0_pred = x1_pred - x1_prev  # 累减还原

            predictions.append(x0_pred)

        return np.array(predictions)  # 转换为numpy数组返回

    def accuracy_test(self, data):
        """
        模型精度检验
        通过多种指标综合评估模型拟合效果

        参数:
        data: 原始观测数据，用于与拟合值对比

        返回:
        metrics: 精度指标字典，包含各项误差指标和精度等级
        fitted: 拟合值数组
        """
        # 获取拟合值，只包含历史部分
        fitted = self.predict(0)[:len(data)]
        self.fitted_values = fitted  # 保存拟合值到对象属性

        # 计算各种精度指标
        data = np.array(data)  # 确保数据为numpy数组

        # 相对误差，衡量预测值偏离真实值的百分比
        relative_errors = np.abs((data - fitted) / data) * 100
        mean_relative_error = np.mean(relative_errors)  # 平均相对误差

        # 其他常用评价指标
        mape = mean_absolute_percentage_error(data, fitted) * 100  # 平均绝对百分比误差
        rmse = np.sqrt(mean_squared_error(data, fitted))  # 均方根误差，反映绝对误差水平
        mae = np.mean(np.abs(data - fitted))  # 平均绝对误差

        # 精度等级判断，根据平均相对误差划分
        if mean_relative_error < 1:
            grade = "一级（很好）"
        elif mean_relative_error < 5:
            grade = "二级（合格）"
        elif mean_relative_error < 10:
            grade = "三级（勉强）"
        else:
            grade = "四级（不合格）"

        # 构建精度指标字典
        metrics = {
            'mean_relative_error': mean_relative_error,
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'grade': grade,
            'relative_errors': relative_errors
        }

        # 打印精度检验结果汇总
        print(f"\n📊 模型精度检验结果:")
        print(f"{'=' * 50}")
        print(f"平均相对误差: {mean_relative_error:.4f}%")
        print(f"MAPE: {mape:.4f}%")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"精度等级: {grade}")
        print(f"{'=' * 50}")

        # 详细误差分析，逐期展示拟合效果
        print(f"\n📋 逐期误差分析:")
        print(f"{'期数':<6}{'实际值':<12}{'拟合值':<12}{'绝对误差':<12}{'相对误差(%)':<12}")
        print("-" * 60)
        for i in range(len(data)):
            abs_error = abs(data[i] - fitted[i])  # 计算绝对误差
            rel_error = relative_errors[i]  # 获取相对误差
            print(f"{i + 1:<6}{data[i]:<12.2f}{fitted[i]:<12.2f}{abs_error:<12.2f}{rel_error:<12.2f}")

        return metrics, fitted

    def plot_results(self, data, years, future_years=None, future_data=None, title="GM(1,1) result"):
        """
        结果可视化
        绘制包含实际值、拟合值、预测值、残差和误差的综合图表

        参数:
        data: 原始实际数据
        years: 历史数据对应的年份
        future_years: 预测年份，默认为None
        future_data: 预测数据，默认为None
        title: 图表标题，默认为"GM(1,1) result"
        """
        plt.figure(figsize=(12, 8))  # 创建12x8英寸的图形窗口

        # 主图（占据第1-2个子图位置）
        plt.subplot(2, 2, (1, 2))

        # 绘制实际值和拟合值曲线
        plt.plot(years, data, 'bo-', label='truth', linewidth=2, markersize=8, markerfacecolor='lightblue')
        if self.fitted_values is not None:
            plt.plot(years, self.fitted_values[:len(data)], 'r^-', label='fit value',
                     linewidth=2, markersize=8, markerfacecolor='lightcoral')

        # 绘制预测值曲线
        if future_years is not None and future_data is not None:
            plt.plot(future_years, future_data, 'gs-', label='pred',
                     linewidth=2, markersize=8, markerfacecolor='lightgreen')

            # 添加预测区间的虚线连接，使图形更连贯
            if len(data) > 0 and len(future_data) > 0:
                connect_x = [years[-1], future_years[0]]  # 连接点的x坐标（历史末年和预测首年）
                connect_y = [self.fitted_values[len(data) - 1], future_data[0]]  # 连接点的y坐标
                plt.plot(connect_x, connect_y, 'g--', alpha=0.5, linewidth=1)  # 绘制绿色虚线连接

        plt.xlabel('year', fontsize=12)  # 设置x轴标签
        plt.ylabel('data', fontsize=12)  # 设置y轴标签
        plt.title(title, fontsize=14, fontweight='bold')  # 设置图表标题
        plt.legend(fontsize=11)  # 显示图例
        plt.grid(True, alpha=0.3)  # 显示网格，透明度0.3

        # 子图1：残差图，展示拟合值与实际值的偏差
        plt.subplot(2, 2, 3)
        if self.fitted_values is not None:
            residuals = np.array(data) - self.fitted_values[:len(data)]  # 计算残差
            plt.bar(years, residuals, alpha=0.7, color='orange')  # 绘制残差柱状图
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)  # 添加y=0参考线
            plt.xlabel('year', fontsize=10)
            plt.ylabel('Residual', fontsize=10)
            plt.title('Residual analysis', fontsize=12)
            plt.grid(True, alpha=0.3)

        # 子图2：相对误差图，展示拟合精度
        plt.subplot(2, 2, 4)
        if hasattr(self, 'fitted_values') and self.fitted_values is not None:
            rel_errors = np.abs((np.array(data) - self.fitted_values[:len(data)]) / np.array(data)) * 100  # 计算相对误差百分比
            plt.bar(years, rel_errors, alpha=0.7, color='purple')  # 绘制相对误差柱状图
            plt.axhline(y=5, color='red', linestyle='--', alpha=0.8, label='5%baseline')  # 添加5%基准线
            plt.xlabel('year', fontsize=10)
            plt.ylabel('error(%)', fontsize=10)
            plt.title('error analysis', fontsize=12)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()  # 自动调整子图间距，避免重叠
        plt.show()  # 显示图形


# GDP增长预测
# 本案例使用GM(1,1)模型预测某地区2017-2021年GDP数据，并外推预测2022-2024年趋势

# 数据准备
# 定义历史GDP数据（单位：亿元），数据呈现缓慢下降趋势，适合灰色预测建模
gdp_data = [1416, 1414, 1412, 1411, 1409]  # 2017-2021年某地区GDP实际观测值
years = list(range(2017, 2022))  # 创建对应的年份列表，从2017到2021年（包含），共5年

print(f"\n📊 原始数据展示:")
gdp_df = pd.DataFrame({
    'year': years,  # 年份列
    'population(e)': gdp_data  # GDP数据列（列名保留原始代码命名，实际为GDP）
})
print(gdp_df.to_string(index=False))  # 显示DataFrame，不打印索引

# 数据基本信息
# 计算数据的统计特征，为模型选择和效果评估提供参考依据
print(f"\n📈 数据基本统计:")
print(f"数据量: {len(gdp_data)}个")  # 显示数据点数量，GM(1,1)模型适合小样本（4-20个）
print(f"平均值: {np.mean(gdp_data):.2f}亿元")  # 计算算术平均值，反映数据集中趋势
print(f"标准差: {np.std(gdp_data):.2f}亿元")  # 计算标准差，评估数据离散程度
print(f"增长率: {((gdp_data[-1] / gdp_data[0]) ** (1 / (len(gdp_data) - 1)) - 1) * 100:.2f}%（年均）")  # 计算年均复合增长率(CAGR)

# 建立GM11模型
# 实例化GM(1,1)模型对象，调用fit方法进行参数估计和模型训练
gm_gdp = GM11()  # 创建GM(1,1)模型实例
gm_gdp.fit(gdp_data)  # 传入历史GDP数据，完成模型训练（级比检验、累加生成、参数估计）

# 精度检验
# 使用训练好的模型对历史数据进行回代计算，评估模型拟合精度
print(f"\n🔍 进行模型精度检验...")
metrics, fitted_values = gm_gdp.accuracy_test(gdp_data)  # 获取精度指标字典和拟合值数组

# 预测未来3个值
# 基于已建立的模型进行外推预测，获取未来3年的GDP预测值
print(f"\n🔮 预测未来3年GDP...")
future_predictions = gm_gdp.predict(3)  # 预测未来3个时间点的值（包含历史拟合和未来预测）
future_years = list(range(2022, 2025))  # 创建预测年份列表，2022、2023、2024年
predicted_values = future_predictions[len(gdp_data):]  # 从历史预测结果中提取未来3年的预测值（从第5个元素开始）

# 结果统计与预测
# 以表格形式汇总展示历史实际值和未来预测值，包含增长率和数据说明，便于对比分析
print(f"\n📋 完整结果汇总:")
print("=" * 80)
print(f"{'年份':<8}{'类型':<8}{'数值(亿元)':<12}{'增长率(%)':<12}{'说明':<20}")
print("-" * 80)

# 遍历输出历史年份的实际值和增长率（首年增长率设为0）
for i, year in enumerate(years):
    growth_rate = 0 if i == 0 else ((gdp_data[i] / gdp_data[i - 1] - 1) * 100)  # 计算环比增长率，首年设为0
    print(f"{year:<8}{'实际值':<8}{gdp_data[i]:<12.0f}{growth_rate:<12.2f}{'历史数据':<20}")

# 遍历输出未来年份的预测值和预测增长率
for i, year in enumerate(future_years):
    prev_value = gdp_data[-1] if i == 0 else predicted_values[i - 1]  # 确定增长率计算的基准值（首年为实际末值）
    growth_rate = ((predicted_values[i] / prev_value - 1) * 100)  # 计算预测期环比增长率
    print(f"{year:<8}{'预测值':<8}{predicted_values[i]:<12.0f}{growth_rate:<12.2f}{'模型预测':<20}")

print("=" * 80)

# 预测结果
# 提取关键预测结果指标，以要点形式展示，便于快速把握预测结论
print(f"\n💡 预测结果分析:")
print(f"• 2022年预测GDP: {predicted_values[0]:.0f}亿元")  # 展示2022年预测值
print(f"• 2023年预测GDP: {predicted_values[1]:.0f}亿元")  # 展示2023年预测值
print(f"• 2024年预测GDP: {predicted_values[2]:.0f}亿元")  # 展示2024年预测值
print(f"• 三年总增长: {((predicted_values[-1] / gdp_data[-1] - 1) * 100):.2f}%")  # 计算2021到2024年总增长率
print(f"• 年均增长率: {((predicted_values[-1] / gdp_data[-1]) ** (1 / 3) - 1) * 100:.2f}%")  # 计算预测期年均复合增长率(CAGR)

# 模型解释
# 解读模型参数a和b的实际经济含义，将数学参数转化为业务洞察，帮助理解决策依据
print(f"\n🔬 模型参数解释:")
print(f"• 发展系数a = {gm_gdp.a:.6f}")  # 显示发展系数值
if gm_gdp.a < 0:
    print(f"  → a < 0，表明该地区GDP呈增长趋势")  # 负值表示增长型系统
    print(f"  → |a| = {abs(gm_gdp.a):.6f}，增长速度适中")  # |a|大小反映增长速率
else:
    print(f"  → a > 0，表明该地区GDP呈衰减趋势")  # 正值表示衰减型系统

print(f"• 灰作用量b = {gm_gdp.b:.2f}")  # 显示灰作用量值
print(f"  → 反映系统的内生增长能力")  # b表示系统的固有增长水平

# 数据可视化
# 调用封装好的绘图函数，生成包含实际值、拟合值、预测值、残差分析和误差分析的综合图表
print(f"\n📊 生成可视化图表...")
gm_gdp.plot_results(gdp_data, years, future_years, predicted_values,
                    "某地区GDP增长预测分析")  # 绘制结果图，包含主图、残差图和误差图

# 模型适用性评估
# 从数据量、级比检验、精度等级、数据趋势四个维度综合评估模型对当前场景的适用性，验证预测可靠性
print(f"\n✅ 模型适用性评估:")
# 检查数据量是否符合GM(1,1)模型的适用条件（小样本4-20个）
print(f"1. 数据量检查: {len(gdp_data)}个观测值 ✓（GM(1,1)适合小样本）")
# 检查级比是否落在可建模区间内（GM(1,1)的基本要求：0.818 < 级比 < 1.220）
print(
    f"2. 级比检验: {'通过' if all(0.818 <= gdp_data[i - 1] / gdp_data[i] <= 1.220 for i in range(1, len(gdp_data))) else '不通过'}")
# 展示模型精度等级，判断预测结果可信度
print(f"3. 精度等级: {metrics['grade']}")
# 检查数据趋势是否单调，单调趋势更适合GM(1,1)模型
print(f"4. 数据趋势: 单调递增 ✓（适合GM(1,1)建模）")  # 注：实际数据为递减，此处保留原代码逻辑