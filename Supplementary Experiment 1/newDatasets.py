import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
np.random.seed(42)
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练集数据
train_data = pd.read_csv('datasets/train_dataset.csv')

# 读取测试集数据
test_data = pd.read_csv('datasets/test_dataset_center_1.5_1.csv')

# 筛选测试集中x2大于1.4的部分
test0 = test_data[test_data['x2'] > 1.4].copy()
print(len(test0))
# 创建test1，将test0中所有样本在x2维度上增加0.1
# 创建第一组数据（原始test1）
test1_part1 = test0.copy()
test1_part1['x2'] = test1_part1['x2'] + 0.05

# 创建第二组数据（随机扰动版本）
test1_part2 = test0.copy()
# 为第二组数据添加随机扰动（在x1和x2维度上都添加）
test1_part2['x1'] = test1_part2['x1'] + np.random.uniform(-0.05, 0.05, size=len(test1_part2))
test1_part2['x2'] = test1_part2['x2'] + np.random.uniform(0.05, 0.15, size=len(test1_part2))  # 保持x2增加的趋势

# 创建第三组数据（另一种随机扰动版本）
test1_part3 = test0.copy()
# 为第三组数据添加不同的随机扰动
test1_part3['x1'] = test1_part3['x1'] + np.random.uniform(-0.1, 0.1, size=len(test1_part3))
test1_part3['x2'] = test1_part3['x2'] + np.random.uniform(0.1, 0.2, size=len(test1_part3))  # 更大的x2增加趋势

# 合并三组数据形成最终的test1
test1 = pd.concat([test1_part1, test1_part2, test1_part3], ignore_index=True)
print(len(test1))

# 修改结束
# 获取测试集去掉test0的部分
test_remaining = test_data[test_data['x2'] <= 1.4].copy()

# 创建图形
plt.figure(figsize=(12, 8))

# 画训练集数据（蓝色）
plt.scatter(train_data['x1'], train_data['x2'], c='blue', label='训练集', alpha=0.6)

# 画测试集剩余部分（绿色）
plt.scatter(test_remaining['x1'], test_remaining['x2'], c='green', label='测试集剩余部分', alpha=0.6)

# 画test0（红色）
plt.scatter(test0['x1'], test0['x2'], c='red', label='test0 (x2>1.4)', alpha=0.6)

# 画test1（紫色）
plt.scatter(test1['x1'], test1['x2'], c='purple', label='test1 (x2+0.1)', alpha=0.6)

# 设置图形标题和标签
plt.title('训练集和测试集各部分数据分布')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)

# 设置图形边界为正方形
plt.axis('equal')

# 保存图形
plt.savefig('补充实验/data_distribution.png')

# 显示图形
plt.show()

plt.close()

# 计算真实问题分布 p_q(x)
# 定义函数 y1 和 y2
def compute_y1(x1, x2):
    return np.sqrt(4 * x1**2 + 3 * x2**2)

def compute_y2(x1, x2):
    return np.sqrt(2 * x1**2 + x2**2)

# 为测试集中的每个样本计算 y1 和 y2 值
test_data['y1'] = compute_y1(test_data['x1'], test_data['x2'])
test_data['y2'] = compute_y2(test_data['x1'], test_data['x2'])

# 设置网格参数
grid_size = 100
T_num = 10000  # 总样本数

# 计算y1和y2的边界
y1_min, y1_max = test_data['y1'].min(), test_data['y1'].max()
y2_min, y2_max = test_data['y2'].min(), test_data['y2'].max()

# 创建网格
y1_grid = np.linspace(y1_min, y1_max, grid_size + 1)
y2_grid = np.linspace(y2_min, y2_max, grid_size + 1)

# 初始化概率分布矩阵
p_q = np.zeros((grid_size, grid_size))

# 创建字典存储每个网格中的点索引
grid_points = {}
for i in range(grid_size):
    for j in range(grid_size):
        grid_points[(i, j)] = []

# 统计每个网格中的点数并记录点索引
for i in range(len(test_data)):
    y1_val = test_data['y1'].iloc[i]
    y2_val = test_data['y2'].iloc[i]
    
    # 找到对应的网格索引
    y1_idx = np.digitize(y1_val, y1_grid) - 1
    y2_idx = np.digitize(y2_val, y2_grid) - 1
    
    # 处理边界值：如果等于最大值，将其分配到最后一个网格
    if y1_idx >= grid_size:
        y1_idx = grid_size - 1
    if y2_idx >= grid_size:
        y2_idx = grid_size - 1
    
    # 确保索引在有效范围内
    if 0 <= y1_idx < grid_size and 0 <= y2_idx < grid_size:
        p_q[y1_idx, y2_idx] += 1
        grid_points[(y1_idx, y2_idx)].append(i)  # 记录点索引

# 计算概率分布
p_q = p_q / T_num

# 为test0计算y1和y2值
test0['y1'] = compute_y1(test0['x1'], test0['x2'])
test0['y2'] = compute_y2(test0['x1'], test0['x2'])

# 为test1计算y1和y2值
test1['y1'] = compute_y1(test1['x1'], test1['x2'])
test1['y2'] = compute_y2(test1['x1'], test1['x2'])

# 重新计算test1的概率分布 p_q1
# 计算test1的y1和y2的边界
y1_min_1, y1_max_1 = test1['y1'].min(), test1['y1'].max()
y2_min_1, y2_max_1 = test1['y2'].min(), test1['y2'].max()

# 创建test1的网格
y1_grid_1 = np.linspace(y1_min, y1_max, grid_size + 1)
y2_grid_1 = np.linspace(y2_min, y2_max, grid_size + 1)

# 初始化test1的概率分布矩阵
p_q1 = np.zeros((grid_size, grid_size))

# 创建字典存储test1每个网格中的点索引
grid_points_1 = {}
for i in range(grid_size):
    for j in range(grid_size):
        grid_points_1[(i, j)] = []

# 统计test1每个网格中的点数并记录点索引
for i in range(len(test1)):
    y1_val = test1['y1'].iloc[i]
    y2_val = test1['y2'].iloc[i]
    
    # 找到对应的网格索引
    y1_idx = np.digitize(y1_val, y1_grid_1) - 1
    y2_idx = np.digitize(y2_val, y2_grid_1) - 1
    
    # 处理边界值：如果等于最大值，将其分配到最后一个网格
    if y1_idx >= grid_size:
        y1_idx = grid_size - 1
    if y2_idx >= grid_size:
        y2_idx = grid_size - 1
    
    # 确保索引在有效范围内
    if 0 <= y1_idx < grid_size and 0 <= y2_idx < grid_size:
        p_q1[y1_idx, y2_idx] += 1
        grid_points_1[(y1_idx, y2_idx)].append(i)  # 记录点索引

# 计算test1的概率分布
p_q1 = p_q1 / T_num

# 初始化grid0和grid1矩阵
grid0 = np.zeros((grid_size, grid_size))  # 存储包含test0点的网格
grid1 = np.zeros((grid_size, grid_size))  # 存储包含test1点的网格

# 标记包含test0点的网格
for i in range(len(test0)):
    y1_val = test0['y1'].iloc[i]
    y2_val = test0['y2'].iloc[i]
    
    # 找到对应的网格索引
    y1_idx = np.digitize(y1_val, y1_grid) - 1
    y2_idx = np.digitize(y2_val, y2_grid) - 1
    
    # 处理边界值
    if y1_idx >= grid_size:
        y1_idx = grid_size - 1
    if y2_idx >= grid_size:
        y2_idx = grid_size - 1
    
    # 确保索引在有效范围内
    if 0 <= y1_idx < grid_size and 0 <= y2_idx < grid_size:
        grid0[y1_idx, y2_idx] = 1  # 标记包含test0点的网格

# 标记包含test1点的网格
for i in range(len(test1)):
    y1_val = test1['y1'].iloc[i]
    y2_val = test1['y2'].iloc[i]
    
    # 找到对应的网格索引
    y1_idx = np.digitize(y1_val, y1_grid_1) - 1
    y2_idx = np.digitize(y2_val, y2_grid_1) - 1
    
    # 处理边界值
    if y1_idx >= grid_size:
        y1_idx = grid_size - 1
    if y2_idx >= grid_size:
        y2_idx = grid_size - 1
    
    # 确保索引在有效范围内
    if 0 <= y1_idx < grid_size and 0 <= y2_idx < grid_size:
        grid1[y1_idx, y2_idx] = 1  # 标记包含test1点的网格

# 从每个网格中随机选取一个点作为新测试集
selected_points = []
selected_probabilities = []  # 存储选中点的概率
selected_grid0_flag = []  # 存储选中点是否来自grid0的标识
for i in range(grid_size):
    for j in range(grid_size):
        if len(grid_points[(i, j)]) > 0:  # 如果网格非空
            # 随机选取一个点
            selected_idx = np.random.choice(grid_points[(i, j)])
            selected_points.append(selected_idx)
            # 记录该网格的概率
            selected_probabilities.append(p_q[i, j])
            # 记录该点是否来自grid0网格 (0表示是，-1表示否)
            selected_grid0_flag.append(0 if grid0[i, j] == 1 else -1)

# 创建新的测试集
new_test_data = test_data.iloc[selected_points].copy()
# 添加概率列
new_test_data['probability'] = selected_probabilities
# 添加grid0标识列
new_test_data['grid0_flag'] = selected_grid0_flag

# 保存新的测试集到CSV文件
new_test_data.to_csv('补充实验/new_test_dataset.csv', index=False)

# 为test1创建新的数据集
test1_selected_points = []
test1_selected_probabilities = []  # 存储选中点的概率
for i in range(grid_size):
    for j in range(grid_size):
        if grid1[i, j] == 1:  # 如果是包含test1点的网格
            if len(grid_points_1[(i, j)]) > 0:  # 如果网格非空
                # 随机选取一个点
                selected_idx = np.random.choice(grid_points_1[(i, j)])
                test1_selected_points.append(selected_idx)
                # 记录该网格的概率
                test1_selected_probabilities.append(p_q1[i, j])

# 创建test1的新数据集
test1_new_data = test1.iloc[test1_selected_points].copy()
# 添加概率列
test1_new_data['probability'] = test1_selected_probabilities

# 保存test1的新数据集到CSV文件
test1_new_data.to_csv('补充实验/test1_new_dataset.csv', index=False)

# 绘制网格图
plt.figure(figsize=(10, 8))

# 绘制不包含test0点但概率不为0的网格（绿色）
for i in range(grid_size):
    for j in range(grid_size):
        # 只绘制概率不为0且不包含test0点的网格
        if p_q[i, j] > 0 and grid0[i, j] == 0:  # 概率不为0且不包含test0点的网格
            y1_start, y1_end = y1_grid[i], y1_grid[i+1]
            y2_start, y2_end = y2_grid[j], y2_grid[j+1]
            plt.gca().add_patch(plt.Rectangle((y1_start, y2_start), 
                                              y1_end - y1_start, y2_end - y2_start,
                                              fill=True, color='green', alpha=0.3))

# 绘制包含test0点的网格（黄色）
for i in range(grid_size):
    for j in range(grid_size):
        if grid0[i, j] == 1:  # 包含test0点的网格
            y1_start, y1_end = y1_grid[i], y1_grid[i+1]
            y2_start, y2_end = y2_grid[j], y2_grid[j+1]
            plt.gca().add_patch(plt.Rectangle((y1_start, y2_start), 
                                              y1_end - y1_start, y2_end - y2_start,
                                              fill=True, color='yellow', alpha=0.6))

# 绘制包含test1点的网格（青色）
for i in range(grid_size):
    for j in range(grid_size):
        if grid1[i, j] == 1:  # 包含test1点的网格
            y1_start, y1_end = y1_grid_1[i], y1_grid_1[i+1]
            y2_start, y2_end = y2_grid_1[j], y2_grid_1[j+1]
            plt.gca().add_patch(plt.Rectangle((y1_start, y2_start), 
                                              y1_end - y1_start, y2_end - y2_start,
                                              fill=True, color='cyan', alpha=0.6))

# 设置图形标题和标签
plt.title('网格分布图')
plt.xlabel('y1')
plt.ylabel('y2')

# 设置坐标轴范围
plt.xlim(min(y1_min, y1_min_1), max(y1_max, y1_max_1))
plt.ylim(min(y2_min, y2_min_1), max(y2_max, y2_max_1))

# 添加图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='yellow', alpha=0.6, label='包含test0点的网格'),
                   Patch(facecolor='cyan', alpha=0.6, label='包含test1点的网格'),
                   Patch(facecolor='green', alpha=0.3, label='概率不为0的网格')]
plt.legend(handles=legend_elements, loc='upper right')

# 保存图形
plt.savefig('补充实验/grid_distribution.png')

# 显示图形
plt.show()

# 创建新文件夹用于存储点及其网格边界信息
points_info_dir = '补充实验/points_with_grid_info'
if not os.path.exists(points_info_dir):
    os.makedirs(points_info_dir)

# 记录new_test_data中每个点及其对应网格的边界信息
points_info_list = []
for idx, (point_idx, prob, grid0_flag) in enumerate(zip(selected_points, selected_probabilities, selected_grid0_flag)):
    # 获取原始点的坐标
    original_point = test_data.iloc[point_idx]
    
    # 计算对应的网格索引
    y1_val = original_point['y1']
    y2_val = original_point['y2']
    y1_idx = np.digitize(y1_val, y1_grid) - 1
    y2_idx = np.digitize(y2_val, y2_grid) - 1
    
    # 处理边界值
    if y1_idx >= grid_size:
        y1_idx = grid_size - 1
    if y2_idx >= grid_size:
        y2_idx = grid_size - 1
    
    # 获取网格边界
    y1_start = y1_grid[y1_idx]
    y1_end = y1_grid[y1_idx + 1]
    y2_start = y2_grid[y2_idx]
    y2_end = y2_grid[y2_idx + 1]
    
    # 记录信息
    points_info_list.append({
        'point_index': point_idx,
        'x1': original_point['x1'],
        'x2': original_point['x2'],
        'y1': y1_val,
        'y2': y2_val,
        'grid_y1_start': y1_start,
        'grid_y1_end': y1_end,
        'grid_y2_start': y2_start,
        'grid_y2_end': y2_end,
        'grid_y1_idx': y1_idx,
        'grid_y2_idx': y2_idx,
        'probability': prob,
        'grid0_flag': grid0_flag,
        'contains_test0': grid0[y1_idx, y2_idx] == 1
    })

# 保存new_test_data的点信息到CSV
points_info_df = pd.DataFrame(points_info_list)
points_info_df.to_csv(os.path.join(points_info_dir, 'new_test_points_with_grid_info.csv'), index=False)

# 记录test1_new_data中每个点及其对应网格的边界信息
test1_points_info_list = []
for idx, (point_idx, prob) in enumerate(zip(test1_selected_points, test1_selected_probabilities)):
    # 获取原始点的坐标
    original_point = test1.iloc[point_idx]
    
    # 计算对应的网格索引
    y1_val = original_point['y1']
    y2_val = original_point['y2']
    y1_idx = np.digitize(y1_val, y1_grid_1) - 1
    y2_idx = np.digitize(y2_val, y2_grid_1) - 1
    
    # 处理边界值
    if y1_idx >= grid_size:
        y1_idx = grid_size - 1
    if y2_idx >= grid_size:
        y2_idx = grid_size - 1
    
    # 获取网格边界
    y1_start = y1_grid_1[y1_idx]
    y1_end = y1_grid_1[y1_idx + 1]
    y2_start = y2_grid_1[y2_idx]
    y2_end = y2_grid_1[y2_idx + 1]
    
    # 记录信息
    test1_points_info_list.append({
        'point_index': point_idx,
        'x1': original_point['x1'],
        'x2': original_point['x2'],
        'y1': y1_val,
        'y2': y2_val,
        'grid_y1_start': y1_start,
        'grid_y1_end': y1_end,
        'grid_y2_start': y2_start,
        'grid_y2_end': y2_end,
        'grid_y1_idx': y1_idx,
        'grid_y2_idx': y2_idx,
        'probability': prob,
        'contains_test1': grid1[y1_idx, y2_idx] == 1
    })

# 保存test1_new_data的点信息到CSV
test1_points_info_df = pd.DataFrame(test1_points_info_list)
test1_points_info_df.to_csv(os.path.join(points_info_dir, 'test1_points_with_grid_info.csv'), index=False)

print(f"点及其网格边界信息已保存到 {points_info_dir} 目录下")
print(f"共记录了 {len(points_info_list)} 个new_test_data的点和 {len(test1_points_info_list)} 个test1_new_data的点")










