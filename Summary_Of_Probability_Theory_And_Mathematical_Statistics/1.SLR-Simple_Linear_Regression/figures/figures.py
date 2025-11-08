# Power by ChatGPT

# figures.py
"""
在 Windows 下生成四张示例图并保存到当前脚本目录；
自动尝试设置中文字体以避免中文乱码。

文件名：
 - univariate_linear.png
 - univariate_nonlinear.png
 - bivariate_linear.png
 - bivariate_nonlinear.png
"""

import os
import sys
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -------------------------
# 1) 设定保存目录（当前脚本所在目录）
# -------------------------
# 在交互式解释器（比如某些 IDE 的 Run）中 __file__ 可能不存在，
# 因此做兼容处理：优先使用 __file__，否则使用当前工作目录。
if getattr(sys, 'frozen', False):
    # 如果使用 PyInstaller 打包后运行，__file__ 可能不可用，使用 exe 路径
    base_path = os.path.dirname(sys.executable)
else:
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_path = os.getcwd()

# -------------------------
# 2) 尝试自动选择系统中的中文字体
# -------------------------
# 优先级字体列表（按顺序查找）
candidate_fonts = [
    "Microsoft YaHei",     # 微软雅黑（Windows 常见）
    "SimHei",              # 黑体（Windows/某些中文环境）
    "Noto Sans CJK SC",    # 谷歌思源（若已安装）
    "Arial Unicode MS",    # 含大量 Unicode 的字体（若安装）
]

# 收集系统已安装字体名（可能很多）
installed_font_names = {f.name for f in fm.fontManager.ttflist}

selected_font = None
for name in candidate_fonts:
    if name in installed_font_names:
        selected_font = name
        break

if selected_font:
    # 设置 matplotlib 全局字体
    matplotlib.rcParams['font.sans-serif'] = [selected_font]
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示问题
    print(f"已找到并设置中文字体：{selected_font}")
else:
    # 未找到上述常见中文字体 — 保留默认但提示用户
    matplotlib.rcParams['axes.unicode_minus'] = False
    warnings.warn(
        "未在系统中找到常见中文字体（Microsoft YaHei / SimHei / Noto Sans CJK SC / Arial Unicode MS）。\n"
        "如果需要避免中文乱码，请在 Windows 上安装「Microsoft YaHei」或「SimHei」，\n"
        "或手动把支持中文的 .ttf 放到系统字体目录并重启 Python。"
    )

# -------------------------
# 固定随机种子以确保可复现
# -------------------------
np.random.seed(0)

# 一个帮助函数：保存并打印路径
def save_fig(fig_or_plt, filename):
    path = os.path.join(base_path, filename)
    # fig_or_plt 可以是 plt（模块）或 figure 对象
    try:
        if hasattr(fig_or_plt, 'savefig'):
            fig_or_plt.savefig(path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(path, dpi=150, bbox_inches='tight')
    except Exception as e:
        # 兼容性保护：若保存时报错，尝试关闭并再试一次
        warnings.warn(f"保存图片时发生异常：{e}")
        plt.savefig(path, dpi=150)
    print(f"✅ 已保存: {path}")
    return path

# -------------------------
# 1) 一元线性示例（散点 + 拟合直线）
#    y = 2x + 1 + 噪声
# -------------------------
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(scale=2.0, size=x.shape)

plt.figure(figsize=(6, 4))
plt.scatter(x, y, label='观测点')
coef = np.polyfit(x, y, 1)
y_fit = np.polyval(coef, x)
plt.plot(x, y_fit, label=f'拟合直线: y={coef[0]:.2f}x+{coef[1]:.2f}')
plt.title('一元线性示例: y ≈ 2x + 1（含噪声）')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
save_fig(plt, 'univariate_linear.png')
plt.close()

# -------------------------
# 2) 一元非线性示例（抛物线 + 噪声）
#    y = x^2 + 噪声
# -------------------------
x = np.linspace(-5, 5, 400)
y = x**2 + np.random.normal(scale=3.0, size=x.shape)

plt.figure(figsize=(6, 4))
plt.scatter(x, y, s=12, label='观测点')
# 画理论抛物线，排序 x 保持线条连续
ix = np.argsort(x)
plt.plot(x[ix], x[ix]**2, label='理论曲线: y=x^2')
plt.title('一元非线性示例: y = x²（含噪声）')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
save_fig(plt, 'univariate_nonlinear.png')
plt.close()

# -------------------------
# 3) 二元线性示例（3D 散点）
#    y ≈ 1.5*x1 + 2.0*x2 + 噪声
# -------------------------
n = 800
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)
y = 1.5 * x1 + 2.0 * x2 + np.random.normal(scale=3.0, size=n)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, s=10)
ax.set_title('二元线性示例: y ≈ 1.5x₁ + 2.0x₂（含噪声）')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('y')
plt.tight_layout()
save_fig(fig, 'bivariate_linear.png')
plt.close()

# -------------------------
# 4) 二元非线性示例（3D 散点）
#    y = sin(x1) + x2^2 + 噪声
# -------------------------
n = 1200
x1 = np.random.uniform(-3, 3, n)
x2 = np.random.uniform(-3, 3, n)
y = np.sin(x1) + x2**2 + np.random.normal(scale=0.5, size=n)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, s=8)
ax.set_title('二元非线性示例: y = sin(x₁) + x₂²（含噪声）')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('y')
plt.tight_layout()
save_fig(fig, 'bivariate_nonlinear.png')
plt.close()

print("\n🎉 所有图片已生成（或已尝试生成）。如果仍有中文乱码，请按下方建议操作。")

# -------------------------
# 小提示（若仍乱码）
# -------------------------
if not selected_font:
    print("\n=== 建议（若仍然出现中文乱码） ===")
    print("1) 在 Windows 上安装常见中文字体（推荐）：Microsoft YaHei（微软雅黑）或 SimHei（黑体）。")
    print("   安装方法：将 .ttf 文件右键安装，或从 Windows 更新/设置 -> 字体 添加。")
    print("2) 重启你的 Python 解释器 / IDE（如 PyCharm），然后再次运行脚本。")
    print("3) 也可以把你想用的中文字体的完整 .ttf 文件路径写入代码，")
    print("   并用 fm.fontManager.addfont(r'完整路径.ttf') 然后手动设置 rcParams 指向该字体。")
    print("示例：")
    print("   fm.fontManager.addfont(r'C:\\path\\to\\your\\SimHei.ttf')")
    print("   matplotlib.rcParams['font.sans-serif'] = ['SimHei']")

