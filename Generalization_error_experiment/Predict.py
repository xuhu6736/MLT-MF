import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from model import FeedForwardNN
from torch.utils.data import TensorDataset

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 检查是否有 GPU 可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 创建保存结果的文件夹
os.makedirs('./泛化误差实验/result', exist_ok=True)

# 定义不同的训练步数
training_steps = [10,50,100,200,500,1000,4000,7000]

# 存储所有训练步数的误差数据
all_errors = {}

# 读取完整的训练数据集
train_data_path = './datasets/train_dataset.csv'
if not os.path.exists(train_data_path):
    print(f'训练数据文件不存在: {train_data_path}')
    exit()

train_data = pd.read_csv(train_data_path)

for step in training_steps:
    # 模型路径
    model_path = f'./泛化误差实验/endpoint/model_step_{step}_checkpoint.pth'
    if not os.path.exists(model_path):
        print(f'模型文件不存在: {model_path}')
        continue

    # 从训练集中取前step个数据
    if len(train_data) < step:
        print(f'训练数据不足{step}个样本')
        continue
    
    # 取前step个数据
    subset_data = train_data.iloc[:step]
    
    # 随机选择10个样本
    if len(subset_data) < 10:
        print(f'数据子集不足10个样本')
        continue
    
    sampled_data = subset_data.sample(n=10, random_state=42)  # 重叠固定随机种子确保可复现
    #sampled_data = train_data.iloc[training_steps[-2]:training_steps[-1]].sample(n=10, random_state=42)#重叠近邻
    # 准备输入数据
    X_sample = torch.tensor(sampled_data[['x1', 'x2']].values, dtype=torch.float32).to(device)
    Y_true = sampled_data[['y1', 'y2']].values

    # 加载模型
    model = FeedForwardNN(learning_rate=0.01).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 进行预测
    with torch.no_grad():
        Y_pred = model(X_sample).cpu().numpy()

    # 计算每个样本的RMSE
    sample_rmse = np.sqrt(np.mean((Y_pred - Y_true) ** 2, axis=1))
    all_errors[step] = sample_rmse
    print(f'训练步数={step} 的10个样本RMSE计算完成')

# 绘制误差曲线图
plt.figure(figsize=(10, 6))
for step in list(all_errors.keys())[:-1]:#重叠近邻
    errors = all_errors[step]
#for step, errors in all_errors.items():#重叠
    plt.plot(range(1, 11), errors, marker='o', label=f'训练步数={step}')


plt.xlabel('样本点索引')
plt.ylabel('均方根误差 (RMSE)')
plt.title('不同训练步数模型的重叠样本预测误差对比')
plt.xticks(range(1, 11))
plt.legend()
plt.grid(True)

# 保存图表
output_path = './泛化误差实验/result/sample_error_comparison.png'
plt.savefig(output_path, dpi=300)
plt.show()
plt.close()
print(f'误差对比图已保存到: {output_path}')