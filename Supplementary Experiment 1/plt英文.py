import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.path import Path
import matplotlib.patches as patches

# 设置Arial字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建图形和坐标轴 - 设置为正方形
fig, ax = plt.subplots(figsize=(8, 6))  # 设置宽高相等，形成正方形

# 设置坐标轴范围 - 确保x和y的范围相同
ax.set_xlim(0.25, 2.25)
ax.set_ylim(0.25, 1.8)

# 确保坐标轴比例相等，防止图形变形
ax.set_aspect('equal', adjustable='box')

# 设置坐标轴标签
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# 设置标题
ax.set_title('Distribution of Training Data and Noisy Test Data')

# 1. 绘制0.5-1.5, 0.5-1.5的蓝色矩形
blue_rect = Rectangle((0.5, 0.5), 1.0, 1.0, color='blue', alpha=0.5, label='Training Data Region')
ax.add_patch(blue_rect)

# 2. 创建圆形路径（以1.5,1为圆心，0.5为半径）
circle_center = (1.5, 1.0)
circle_radius = 0.5

# 3. 绘制大部分为红色的圆
# 创建一个圆，但排除x2>1.4的部分
# 生成圆上的点
theta = np.linspace(0, 2*np.pi, 1000)
x_circle = circle_center[0] + circle_radius * np.cos(theta)
y_circle = circle_center[1] + circle_radius * np.sin(theta)

# 创建红色圆（排除x2>1.4的部分）
red_mask = y_circle <= 1.4
ax.fill(x_circle[red_mask], y_circle[red_mask], 'red', alpha=0.5, label='Test Set Region')

# 4. 绘制x2>1.4的绿色部分
green_mask = y_circle > 1.4
#ax.fill(x_circle[green_mask], y_circle[green_mask], 'white', alpha=0.5, label='Excluded Region')

# 5. 将绿色部分复制并向上平移0.5后为黄色
# 计算绿色部分的点并平移
x_green_part = x_circle[green_mask]
y_green_part = y_circle[green_mask] + 0.05  # 向上平移0.5

# 绘制深黑色的灰色部分
ax.fill(x_green_part, y_green_part, '#303030', alpha=0.5, label='Noise-Added Region')


# 添加网格
ax.grid(True, linestyle='--', alpha=0.7)

# 添加图例
ax.legend(loc='upper right')

# 保存图形
plt.tight_layout()
plt.savefig(r'c:\Users\和军训啊\Desktop\代码和数据集\补充实验\data_distribution.png', dpi=300)

# 显示图形
plt.show()