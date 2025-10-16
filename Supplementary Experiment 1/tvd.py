import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from model import FeedForwardNN
import matplotlib.font_manager as fm

# 设置Arial字体支持
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用Arial字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查是否有 GPU 可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 读取数据并提取未覆盖区域的点
def get_uncovered_data():
    # 读取new_test_dataset.csv
    dataset_path = r'补充实验\new_test_dataset.csv'
    test_df = pd.read_csv(dataset_path)
    
    # 读取test1_new_dataset.csv
    test1_dataset_path = r'补充实验\test1_new_dataset.csv'
    test1_df = pd.read_csv(test1_dataset_path)
    
    # 创建新的噪声输入信息分布数组
    noise_input_df = test_df[test_df['grid0_flag'] == -1]
    # 合并noise_input_df和test1_df
    combined_df = pd.concat([noise_input_df, test1_df])
    
    # 筛选出未覆盖区域的点 (x1和x2不在0.5~1.5范围内)
    uncovered_df = combined_df[    
        (combined_df['x1'] < 0.5) | (combined_df['x1'] > 1.5) |
        (combined_df['x2'] < 0.5) | (combined_df['x2'] > 1.5)
    ].copy()
    
    return uncovered_df

# 加载模型
def load_model(model_path):
    # 创建模型实例
    model = FeedForwardNN(learning_rate=0.01).to(device)
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 设置为评估模式
    model.eval()
    
    return model

# 读取tvd_results.txt中的5.17公式右侧上限值
def get_formula_upper_bound():
    results_file = r'补充实验\tvd_results.txt'
    if not os.path.exists(results_file):
        print(f'结果文件不存在: {results_file}')
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if '5.17公式右侧上限:' in line:
                    upper_bound = float(line.split(':', 1)[1].strip())
                    return upper_bound
        print('未找到5.17公式右侧上限值')
        return None
    except Exception as e:
        print(f'读取结果文件时出错: {e}')
        return None

# 主函数
def main():
    # 获取未覆盖区域的数据
    uncovered_df = get_uncovered_data()
    
    if uncovered_df.empty:
        print("未覆盖区域为空，无法进行预测")
        return
    
    # 随机选择100组数据，如果数据少于100组则选择所有数据
    sample_size = min(100, len(uncovered_df))
    random_sample = uncovered_df.sample(n=sample_size, random_state=42)
    
    # 准备输入数据 (x1, x2)
    X_input = torch.tensor(random_sample[['x1', 'x2']].values, dtype=torch.float32).to(device)
    # 获取真实值 (y1, y2)
    Y_true = random_sample[['y1', 'y2']].values
    
    # 加载模型
    model_path = r'泛化误差实验\endpoint\model_step_20_checkpoint.pth'
    
    if not os.path.exists(model_path):
        print(f'模型文件不存在: {model_path}')
        return
    
    model = load_model(model_path)
    
    # 进行预测
    with torch.no_grad():
        Y_pred = model(X_input).cpu().numpy()
    
    # 计算每个样本的RMSE
    sample_rmse = np.sqrt(np.mean((Y_pred - Y_true) ** 2, axis=1))
    
    # 计算总体RMSE
    total_rmse = np.sqrt(np.mean((Y_pred - Y_true) ** 2))
    
    # 输出结果
    print(f"预测样本数量: {sample_size}")
    print(f"总体RMSE: {total_rmse:.4f}")

    
    # 读取5.17公式右侧上限值
    formula_upper_bound = get_formula_upper_bound()
    if formula_upper_bound is not None:
        print(f"5.17公式右侧上限值: {formula_upper_bound:.6f}")
    
    # 保存预测结果和RMSE
    prediction_results = pd.DataFrame({
        'x1': random_sample['x1'].values,
        'x2': random_sample['x2'].values,
        'y1_true': Y_true[:, 0],
        'y2_true': Y_true[:, 1],
        'y1_pred': Y_pred[:, 0],
        'y2_pred': Y_pred[:, 1],
        'rmse': sample_rmse
    })
    
    # 保存结果到CSV文件
    results_file = r'补充实验\prediction_rmse_results.csv'
    prediction_results.to_csv(results_file, index=False)
    print(f"预测结果和RMSE已保存到: {results_file}")
    
    # 使用Predict.py风格的结果显示方式
    plt.figure(figsize=(10, 6))
    
    # 绘制每个样本的RMSE值，横坐标为样本索引
    plt.plot(range(1, len(sample_rmse) + 1), sample_rmse, marker='o', label=f'Model training steps=20')
    
    # 如果找到了5.17公式右侧上限值，添加红横线
    if formula_upper_bound is not None:
        plt.axhline(y=formula_upper_bound, color='r', linestyle='-', linewidth=1, 
                   label=f'TVD upper bound of noisy input: {formula_upper_bound:.4f}')
    
    plt.xlabel('Sample index')
    plt.ylabel('Root Mean Square Error (RMSE)')
    plt.title('Model Error Test Results for Noisy Information Input')
    # 不显示具体的横坐标索引值
    plt.xticks([])
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    output_path = r'补充实验\sample_error_comparison.png'
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()
    print(f'误差对比图已保存到: {output_path}')

if __name__ == "__main__":
    main()