import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

class FeedForwardNN(nn.Module):
    def __init__(self, learning_rate=1e-4):
        super(FeedForwardNN, self).__init__()
        
        # 网络层定义
        self.hidden = nn.Linear(2, 3)  # 输入层(2)→隐藏层(3)
        self.output = nn.Linear(3, 2)   # 隐藏层(3)→输出层(2)，输出y1,y2
        
        # 激活函数x
        self.relu = nn.ReLU()

        # 优化器和损失函数
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()  # 使用均方误差损失
    
    def forward(self, X):
        # 前向传播
        hidden = self.relu(self.hidden(X))  # ReLU激活
        output = self.output(hidden) #恒等激活
        return output 
        
    def predict(self, X):
        """预测回归值(y1,y2)"""
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            return self(X).cpu().numpy()
