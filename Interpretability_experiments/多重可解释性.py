import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from model import FeedForwardNN
import glob
import re
import matplotlib.animation as animation
# 调大所有图的标识字号
plt.rcParams["font.size"] = 14  # 全局默认字体大小
plt.rcParams["axes.titlesize"] = 16  # 图表标题字体大小
plt.rcParams["axes.labelsize"] = 14  # 坐标轴标签字体大小
plt.rcParams["xtick.labelsize"] = 12  # x轴刻度字体大小
plt.rcParams["ytick.labelsize"] = 12  # y轴刻度字体大小
plt.rcParams["legend.fontsize"] = 14  # 图例字体大小
# 设置字体为 Arial
plt.rcParams["font.family"] = "Arial"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
np.random.seed(42)
# 创建结果文件夹
os.makedirs('./可解释性实验/result', exist_ok=True)

# 功能一：绘制训练集和测试集数据分布，并标记预测结果
# 功能一：为第1到50步检查点生成动画视频
def plot_data_distribution():    
    # 读取数据集
    train_file = './datasets/train_dataset.csv'
    test_file = './datasets/test_dataset_center_-1_-1.csv'
    
    if not os.path.exists(train_file):
        print(f'找不到训练集文件 {train_file}')
        return
    if not os.path.exists(test_file):
        print(f'找不到测试集文件 {test_file}')
        return
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    print(f'训练集样本数: {len(train_df)}')
    print(f'测试集样本数: {len(test_df)}')
    
    # 生成随机点
    num_random_points = 20000
    x1_random = np.random.uniform(-2, 4, num_random_points)
    x2_random = np.random.uniform(-2, 4, num_random_points)
    random_points = np.column_stack((x1_random, x2_random))
    
    # 确保随机点不在指定范围内
    valid_random_points = []
    for point in random_points:
        x, y = point
        in_rectangle = (0.5 <= x <= 1.5) and (0.5 <= y <= 1.5)
        distance_from_center = np.sqrt((x+1)**2 + (y+1)**2)
        in_circle = distance_from_center <= 0.5
        if not in_rectangle and not in_circle:
            valid_random_points.append(point)
    
    valid_random_points = np.array(valid_random_points)
    print(f'生成的有效随机点数: {len(valid_random_points)}')
    
    # 准备数据
    train_points = train_df[['x1', 'x2']].values
    test_points = test_df[['x1', 'x2']].values
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制禁止区域（只绘制一次）
    rectangle = plt.Rectangle((0.5, 0.5), 1.0, 1.0, facecolor='gray', alpha=0.2)
    circle = plt.Circle((-1, -1), 0.5, facecolor='gray', alpha=0.2)
    
    def animate(step):
        ax.clear()
        
        # 检查检查点文件是否存在
        checkpoint_path = f'./可解释性实验/endpoint/model_step_{step}_checkpoint.pth'
        if not os.path.exists(checkpoint_path):
            print(f'找不到检查点文件 {checkpoint_path}')
            return
        
        # 加载模型
        model = FeedForwardNN()
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f'加载检查点 {step} 时出错: {str(e)}')
            return
        
        # 对数据进行预测
        def predict_points(points):
            if len(points) == 0:
                return np.array([])
            X_tensor = torch.FloatTensor(points)
            with torch.no_grad():
                predictions = model.predict(X_tensor)
            return np.array([0 if (pred[0] < 0 or pred[1] < 0) else 1 for pred in predictions])
        
        # 预测训练集
        if len(train_points) > 0:
            train_pred = predict_points(train_points)
            train_positive = train_points[train_pred == 1]
            train_negative = train_points[train_pred == 0]
            if len(train_positive) > 0:
                ax.scatter(train_positive[:, 0], train_positive[:, 1], alpha=0.7, color='slateblue', s=40, label='Train Datasets - Positive Prediction')
            if len(train_negative) > 0:
                ax.scatter(train_negative[:, 0], train_negative[:, 1], alpha=0.7, color='#C62828', s=40, label='Train Datasets - Negative Prediction')
        
        # 预测测试集
        if len(test_points) > 0:
            test_pred = predict_points(test_points)
            test_positive = test_points[test_pred == 1]
            test_negative = test_points[test_pred == 0]
            if len(test_positive) > 0:
                ax.scatter(test_positive[:, 0], test_positive[:, 1], alpha=0.7, color='g', s=40, label='Test Datasets - Positive Prediction')
            if len(test_negative) > 0:
                ax.scatter(test_negative[:, 0], test_negative[:, 1], alpha=0.7, color='red', s=40, label='Test Datasets - Negative Prediction')
        
        # 预测随机点
        if len(valid_random_points) > 0:
            random_pred = predict_points(valid_random_points)
            random_positive = valid_random_points[random_pred == 1]
            random_negative = valid_random_points[random_pred == 0]
            if len(random_positive) > 0:
                ax.scatter(random_positive[:, 0], random_positive[:, 1], alpha=0.7, color='darkseagreen', s=40, label='Random Points - Positive Prediction')
            if len(random_negative) > 0:
                ax.scatter(random_negative[:, 0], random_negative[:, 1], alpha=0.7, color='darkorange', s=40, label='Random Points - Negative Prediction')
        
        # 添加禁止区域
        ax.add_patch(plt.Rectangle((0.5, 0.5), 1.0, 1.0, facecolor='gray', alpha=0.2))
        ax.add_patch(plt.Circle((-1, -1), 0.5, facecolor='gray', alpha=0.2))
        
        # 设置图形属性
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'Datasets and Random Points Prediction Visualization (Checkpoint at Step {step})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 4)
    
    # 创建动画
    print('开始生成动画...')
    anim = animation.FuncAnimation(fig, animate, frames=range(165,166), interval=500, repeat=True)
    #可修改上述frames进行动画生成
    # 保存为视频
    save_path = './可解释性实验/result/data_prediction_animation.mp4'
    anim.save(save_path, writer='ffmpeg', fps=2, dpi=300)
    plt.close()
    print(f'动画已保存到: {save_path}')

# 功能二：显示检查点的参数
def visualize_model_parameters():
    checkpoint_step = 165
    checkpoint_path = f'./可解释性实验/endpoint/model_step_{checkpoint_step}_checkpoint.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f'找不到检查点文件 {checkpoint_path}')
        return
    
    # 加载模型参数
    model = FeedForwardNN()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # 收集参数
    params = []
    param_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param.data.numpy())
            param_names.append(name)
    
    # 创建保存目录（如果不存在）
    os.makedirs('./可解释性实验/result', exist_ok=True)
    save_path = f'./可解释性实验/result/model_parameters_step_{checkpoint_step}.txt'
    
    # 保存参数信息到txt文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f'Model Parameters Detailed Information - Checkpoint at Step {checkpoint_step}\n')
        f.write('=' * 80 + '\n\n')
        
        for name, param_values in zip(param_names, params):
            # 写入参数信息
            f.write(f'Parameter Name: {name}\n')
            f.write(f'Parameter Shape: {param_values.shape}\n')
            f.write(f'Parameter Count: {param_values.size}\n')
            f.write(f'Parameter Values:\n{param_values}\n')
            f.write('-' * 80 + '\n\n')
    
    print(f'Model parameters detailed information saved to: {save_path}')

# 功能三：绘制前1000步检查点对(1,1)和(-1.3535,-1.3535)的预测值变化
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import re
import os

def plot_error_vs_steps():
    # 定义目标点
    point_1 = np.array([1.0, 1.0])  # (1, 1) 点
    point_2 = np.array([-1.3535, -1.3535])  # 以 (-1, -1) 为圆心，半径为 0.5 的圆的最左下点

    # 计算真实值
    def calculate_true_values(x1, x2):
        y1_true = np.sqrt(4 * x1**2 + 3 * x2**2)
        y2_true = np.sqrt(2 * x1**2 + x2**2)
        return np.array([y1_true, y2_true])

    # 查找所有检查点文件
    checkpoint_files = glob.glob('./可解释性实验/endpoint/model_step_*_checkpoint.pth')

    # 提取步数并筛选 1-400 步
    step_pattern = re.compile(r'model_step_(\d+)_checkpoint.pth')
    steps = []

    for file in checkpoint_files:
        match = step_pattern.search(file)
        if match:
            step = int(match.group(1))
            if 1 <= step <= 400:
                steps.append(step)

    steps.sort()
    print(f'找到{len(steps)}个符合条件的检查点')

    # 存储结果 - 分别存储每个点的y1和y2预测值
    pred_values_1_y1 = []  # (1,1)点的y1预测值
    pred_values_1_y2 = []  # (1,1)点的y2预测值
    pred_values_2_y1 = []  # (-1.3535,-1.3535)点的y1预测值
    pred_values_2_y2 = []  # (-1.3535,-1.3535)点的y2预测值
    step_numbers = []

    # 对每个检查点进行预测
    for step in steps:
        checkpoint_path = f'./可解释性实验/endpoint/model_step_{step}_checkpoint.pth'

        try:
            # 加载模型
            model = FeedForwardNN()
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            model.eval()

            # 准备输入
            tensor_1 = torch.FloatTensor([point_1])
            tensor_2 = torch.FloatTensor([point_2])

            # 预测
            with torch.no_grad():
                pred_1 = model.predict(tensor_1)[0]
                pred_2 = model.predict(tensor_2)[0]

            # 存储预测值
            pred_values_1_y1.append(pred_1[0])
            pred_values_1_y2.append(pred_1[1])
            pred_values_2_y1.append(pred_2[0])
            pred_values_2_y2.append(pred_2[1])
            step_numbers.append(step)

            if step % 100 == 0:
                print(f'已处理步数: {step}, (1,1)点 y1: {pred_1[0]:.6f}, y2: {pred_1[1]:.6f}')
                print(f'           (-1.3535,-1.3535)点 y1: {pred_2[0]:.6f}, y2: {pred_2[1]:.6f}')
        except Exception as e:
            print(f'处理步数 {step} 时出错: {str(e)}')
            continue

    # 创建图形
    plt.figure(figsize=(16, 10))

    # 绘制预测值曲线
    plt.plot(step_numbers, pred_values_1_y1, 'b-', linewidth=2, label='Predicted y1 at (1,1) in Training Datasets')
    plt.plot(step_numbers, pred_values_1_y2, 'g-', linewidth=2, label='Predicted y2 at (1,1) in Training Datasets')
    plt.plot(step_numbers, pred_values_2_y1, 'r-', linewidth=2, label='Predicted y1 at (-1.3535,-1.3535) in Test Datasets')
    plt.plot(step_numbers, pred_values_2_y2, 'm-', linewidth=2, label='Predicted y2 at (-1.3535,-1.3535) in Test Datasets')
    
    # 添加散点图
    plt.scatter(step_numbers, pred_values_1_y1, c='blue', s=10)
    plt.scatter(step_numbers, pred_values_1_y2, c='green', s=10)
    plt.scatter(step_numbers, pred_values_2_y1, c='red', s=10)
    plt.scatter(step_numbers, pred_values_2_y2, c='magenta', s=10)

    # 添加红色虚线y=0
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)

    plt.xlabel('Training Steps')
    plt.ylabel('Predicted Values')
    plt.title('Changes in Predicted Values for Two Points with Training Steps')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 保存图形
    save_path = './可解释性实验/result/error_vs_steps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Predicted values change plot saved to: {save_path}')

# 确保在main中调用
if __name__ == '__main__':
    print('=== 执行功能一：绘制数据分布图 ===')
    #可修改上述frames进行动画生成
    plot_data_distribution()
    print('=== 执行功能二：显示检查点的参数 ===')
    visualize_model_parameters()
    print('=== 执行功能三：绘制预测值随步数变化曲线 ===')
    plot_error_vs_steps()
    print('\nAll functions completed, results saved in ./可解释性实验/result folder')