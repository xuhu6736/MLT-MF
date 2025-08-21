import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from model import FeedForwardNN
from torch.utils.data import DataLoader, TensorDataset

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 固定随机种子
seed = 43
np.random.seed(seed)
torch.manual_seed(seed) #CPU
torch.cuda.manual_seed(seed) #GPU

# 检查是否有 GPU 可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 创建保存模型和结果的文件夹
os.makedirs('./泛化误差实验/endpoint', exist_ok=True)
os.makedirs('./泛化误差实验/result', exist_ok=True)

# 加载训练集
train_data = pd.read_csv('./datasets/train_dataset.csv')
print(f'训练集大小: {len(train_data)}')

# 准备训练数据并移至 GPU
X_train = torch.tensor(train_data[['x1', 'x2']].values, dtype=torch.float32).to(device)
Y_train = torch.tensor(train_data[['y1', 'y2']].values, dtype=torch.float32).to(device)

# 设置批大小和数据加载器 (batch_size=1)
batch_size = 1
dataset = TensorDataset(X_train, Y_train)
# 不打乱数据
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 创建模型实例并移至 GPU
model = FeedForwardNN(learning_rate=0.01).to(device)#0.01

# 训练模型
print(f"开始训练模型...")
max_epochs = 1  
step_count = 0

# 用于存储loss值的列表
loss_history = []

for epoch in range(max_epochs): 
    epoch_loss = 0.0
    for batch_X, batch_Y in dataloader:
        # 前向传播
        model.train()
        Y_pred = model(batch_X)
        # 计算损失
        loss = model.loss_fn(Y_pred, batch_Y)
        # 反向传播和优化
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        # 累加损失
        epoch_loss += loss.item()
        # 保存loss值
        loss_history.append(loss.item())
        #print(f"Step {step_count}, Loss: {loss.item():.6f}")
        # 增加步数计数
        step_count += 1
        model_path = f'./泛化误差实验/endpoint/model_step_{step_count}_checkpoint.pth'
        torch.save(model.state_dict(), model_path)

# 训练结束后保存最终模型
final_model_path = './泛化误差实验/endpoint/model_final.pth'
torch.save(model.state_dict(), final_model_path)
print(f"最终模型已保存到: {final_model_path}")

# 绘制loss变化图
plt.figure(figsize=(10, 6))

max_points = min(600, len(loss_history)) #前600个点
plt.plot(range(1, max_points + 1), loss_history[:max_points])
plt.xlabel('训练步数')
plt.ylabel('损失值')
plt.title('训练过程中的损失变化 (前600个点)')
plt.grid(True)

# 保存图表
loss_plot_path = f'./泛化误差实验/result/loss_history_{max_points}.png'
plt.savefig(loss_plot_path, dpi=300)
plt.show()
plt.close()
print(f'损失变化图已保存到: {loss_plot_path}')

        
        