import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from model import FeedForwardNN
import matplotlib.font_manager as fm
# 设置Arial字体
#plt.rcParams['font.sans-serif'] = ['Arial']
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查是否有 GPU 可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 读取数据并根据grid0_flag处理数据
def get_processed_data():
    # 读取new_test_dataset.csv
    new_test_path = r'补充实验\new_test_dataset.csv'
    new_test_df = pd.read_csv(new_test_path)
    
    # 读取test1_new_dataset.csv
    test1_path = r'补充实验\test1_new_dataset.csv'
    test1_df = pd.read_csv(test1_path)
    
    # 随机选择100组数据
    sample_size = min(100, len(new_test_df))
    random_sample = new_test_df.sample(n=sample_size, random_state=42).copy()
    
    # 处理每个样本
    processed_samples = []
    for _, row in random_sample.iterrows():
        if row['grid0_flag'] == 0:
            # grid0_flag=0的情况：将x2+0.1，然后找到最近的点
            x1 = row['x1']
            x2_plus = row['x2'] + 0.1
            
            # 计算与test1_new_dataset中所有点的欧氏距离
            distances = np.sqrt((test1_df['x1'] - x1) ** 2 + (test1_df['x2'] - x2_plus) ** 2)
            # 找到最近的点
            nearest_idx = distances.idxmin()
            nearest_point = test1_df.loc[nearest_idx].copy()
            
            # 使用最近点作为替代输入
            processed_samples.append(nearest_point)
        elif row['grid0_flag'] == -1:
            # grid0_flag=-1的情况：直接使用原始输入
            processed_samples.append(row)
        else:
            # 其他情况也直接使用原始输入
            processed_samples.append(row)
    
    # 将处理后的样本转换为DataFrame
    processed_df = pd.DataFrame(processed_samples)
    
    return processed_df

# 加载模型
def load_model(model_path):
    # 创建模型实例
    model = FeedForwardNN(learning_rate=0.01).to(device)
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 设置为评估模式
    model.eval()
    
    return model

# 读取tvd_results.txt中的5.18公式右侧上限2的值
def get_formula_upper_bound():
    results_file = r'补充实验\tvd_results.txt'
    if not os.path.exists(results_file):
        print(f'结果文件不存在: {results_file}')
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if '5.18公式右侧上限2:' in line:
                    upper_bound = float(line.split(':', 1)[1].strip())
                    return upper_bound
        print('未找到5.18公式右侧上限2的值')
        return None
    except Exception as e:
        print(f'读取结果文件时出错: {e}')
        return None

# 主函数
def main():
    # 获取处理后的数据
    processed_df = get_processed_data()
    
    if processed_df.empty:
        print("处理后的数据为空，无法进行预测")
        return
    
    # 准备输入数据 (x1, x2)
    X_input = torch.tensor(processed_df[['x1', 'x2']].values, dtype=torch.float32).to(device)
    # 获取真实值 (y1, y2)
    Y_true = processed_df[['y1', 'y2']].values
    
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
    print(f"预测样本数量: {len(processed_df)}")
    print(f"总体RMSE: {total_rmse:.4f}")
    
    # 读取5.18公式右侧上限2的值
    formula_upper_bound = get_formula_upper_bound()
    if formula_upper_bound is not None:
        print(f"5.18公式右侧上限2的值: {formula_upper_bound:.6f}")
    
    # 保存预测结果和RMSE
    prediction_results = pd.DataFrame({
        'x1': processed_df['x1'].values,
        'x2': processed_df['x2'].values,
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
    plt.figure(figsize=(10, 8))
    
    # 绘制每个样本的RMSE值，横坐标为样本索引
    plt.plot(range(1, len(sample_rmse) + 1), sample_rmse, marker='o', label=f'模型训练步数=20')
    
    # 如果找到了5.18公式右侧上限2的值，添加红横线
    if formula_upper_bound is not None:
        plt.axhline(y=formula_upper_bound, color='r', linestyle='-', linewidth=1, 
                   label=f'真实信息的TVD上界: {formula_upper_bound:.4f}')
    
    plt.xlabel('样本索引')
    plt.ylabel('均方根误差 (RMSE)')
    plt.title('δ<γ条件下真实输入与模型预测的误差统计')
    # 不显示具体的横坐标索引值
    plt.xticks([])
    # 设定纵轴为4
    plt.ylim(1.5, 4)
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    output_path = r'补充实验\sample_error_comparison2.png'
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()
    print(f'误差对比图已保存到: {output_path}')

if __name__ == "__main__":
    main()