import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import torch
from model import FeedForwardNN

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建结果文件夹
os.makedirs('./伦理实验/result', exist_ok=True)

def calculate_theoretical_error():
    # 读取训练集
    train_file = "./datasets/test_dataset_center_-1_-1.csv"
    if not os.path.exists(train_file):
        print(f'找不到训练集文件 {train_file}')
        return
    
    # 读取训练集
    train_df = pd.read_csv(train_file)
    total_samples = len(train_df)
    print(f'训练集总样本数: {total_samples}')
    
    # 加载模型
    checkpoint_path = './伦理实验/endpoint/model_step_165_checkpoint.pth'
    if not os.path.exists(checkpoint_path):
        print(f"警告：找不到检查点文件 {checkpoint_path}")
        return
        
    # 加载模型
    model = FeedForwardNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    # 准备实验结果表格
    results = []
    
    # 进行10次实验
    for experiment in range(1, 11):
        print(f'\n实验 {experiment}/10:')
        
        # 设置不同的随机种子
        np.random.seed(experiment)
        
        # 随机选取5000个样本
        sample_indices = np.random.choice(total_samples, 5000, replace=False)
        sample_df = train_df.iloc[sample_indices]
        
        # 准备测试数据
        test_input = sample_df[['x1', 'x2']].values
        test_tensor = torch.FloatTensor(test_input)
        
        # 1. 无伦理检查的预测
        print("1. 无伦理检查的预测:")
        # 使用time.perf_counter()获得更高精度
        start_time = time.perf_counter()
        with torch.no_grad():
            predictions = model.predict(test_tensor)
            for i, (y1, y2) in enumerate(predictions):
                print(f"({y1:.3f}, {y2:.3f})")
        end_time = time.perf_counter()
        no_ethics_time = end_time - start_time
        print(f"无伦理检查的预测时间: {no_ethics_time:.6f} 秒")
        
        # 保存预测结果
        sample_df = sample_df.copy() 
        sample_df['pred_y1'] = predictions[:, 0]
        sample_df['pred_y2'] = predictions[:, 1]
        
        # 2. 有伦理检查的预测
        print("\n2. 有伦理检查的预测:")
        start_time = time.perf_counter()
        ethics_violations = 0
        with torch.no_grad():
            predictions = model.predict(test_tensor)
            # 伦理检查
            for i, (y1, y2) in enumerate(predictions):
                if y1 < 0 or y2 < 0:
                    ethics_violations = ethics_violations + 1
                    print("伦理错误")
                else:
                    print(f"({y1:.3f}, {y2:.3f})")

        end_time = time.perf_counter()
        with_ethics_time = end_time - start_time
        print(f"有伦理检查的预测时间: {with_ethics_time:.6f} 秒")
        print(f"伦理错误数量: {ethics_violations}")
        no_ethics_errors = len(sample_df) - ethics_violations
        
        # 计算开销比值
        if with_ethics_time > 0:
            cost_ratio = with_ethics_time / no_ethics_time
        else:
            cost_ratio = float('inf')  # 避免除零错误
        
        # 无伦理错误检测成功率 (恒等100%)
        no_ethics_success_rate = 100.0
        
        # 有伦理错误检测成功率 (恒等100%)
        with_ethics_success_rate = 100.0
        
        # 保存实验结果
        results.append({
            '实验序号': experiment,
            '正常计算时间(秒)': no_ethics_time,
            '有伦理实验的时间(秒)': with_ethics_time,
            '开销比值': cost_ratio,
            '无伦理错误数量': no_ethics_errors,
            '伦理错误数量': ethics_violations,
            '无伦理错误检测成功率(%)': no_ethics_success_rate,
            '有伦理错误检测成功率(%)': with_ethics_success_rate
        })
    
    # 打印结果表格
    print('\n实验结果汇总:')
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 保存结果到文件
    results_file = './伦理实验/result/experiment_results.csv'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(results_df.to_string(index=False))
    print(f'\n实验结果已保存到: {results_file}')

if __name__ == '__main__':
    calculate_theoretical_error()