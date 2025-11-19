# Python可视化

## Matplotlib库函数

figure(figsize(x,y)) 创建画布(分别率为100x,100y)

title("",fontsize=(int)) 设置标题和字体大小

xlabel("",fontsize=(int)) 设置图x轴的名称和字体大小

ylabel("",fontsize=(int)) 设置图y轴的名称和字体大小

plot(X_test, y_pred, color='red', linewidth=2, label='回归线') 绘制直线 可加参数'bo-'或'gs-'绘制节点上为三角形或正方形的线

scatter(X_test, y_test, color='blue', label='实际成绩', alpha=0.7) 绘制数据散点图

axhline(y=0, color='red', linestyle='--') 添加一条y=0的参考线，跨越整个x轴

axvline(x=60, color='gray', linestyle=':', alpha=0.7,label='分界线') 在x=60处添加灰色垂直参考线，linestyle=':'设置为点线

phlines(y=0, xmin=data.min(), xmax=data.max(), colors='red') 添加红色水平参考线，可指定起点和终点

fill_between(future_time, pred_lower, pred_upper, alpha=0.3, color='green',label='95%置信区间') 填充预测区间，fill_between在上下界之间绘制半透明绿色区域，alpha=0.3设置透明度，直观展示预测不确定性范围

bar(years, residuals, alpha=0.7, color='orange')  # 绘制柱状图

legend() 显示示意图,比如说红色的线是什么,蓝色的点是什么，需要在绘制函数中加入lable="lable"参数

grid(True) 设置网格线

## Pandas库函数

DataFrame({'Xdata': xdata.flatten(), 'Ydata': ydata.flatten()}) 将两列数字转换成DataFrame格式方便处理

head() 前五行数据

describe() 描述数据的基本信息

data['Xdata'].corr(data['Ydata']) Xdata与Ydata的相关系数

data_df['TargetVar'] = data.target 设定目标值为TargetVar下的数据,用于确定多元线性分析中的预测目标值

read_csv('') 读取csv文件，参数为相对路径

df['Var'].plot(kind='hist', bins=100) pandas自带的绘图，kind='hist'指定了图表类型为直方图，bins=100 将数据范围（从最小值到最大值）分割成 100 个等宽的“桶”或“区间”

## Seaborn库函数

set_style('whitegrid') 设置可视化风格

scatterplot(x='Xdata', y='Ydata', data=data, alpha=0.7) 绘制散点图(x为Xdata，y为Ydata，data为处理过后的pandas数据，alpha为点的大小，绘制的图为全部数据的散点图)

heatmap(data_df.corr(), annot=True, cmap='coolwarm', fmt=".2f") # corr()函数为数据的相关系数，annot为是否显示具体数据，cmap表示映射的方案，fmt为数据保留几位小数

histplot(df['FloodProbability'], bins=100, kde=True) 绘制直方图,并在柱状图上叠加了蓝色KDE核密度估计曲线，更平滑地展示数据分布趋势

catplot(x='Xdata',y='Ydata',data=data, kind='point') 绘制点图
