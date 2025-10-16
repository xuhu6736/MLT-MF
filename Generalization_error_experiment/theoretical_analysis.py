import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
from model import FeedForwardNN
import math

# 设置字体为 Arial
plt.rcParams["font.family"] = "Arial"
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of negative sign display

# 调大所有图的标识字号
plt.rcParams["font.size"] = 14  # 全局默认字体大小
plt.rcParams["axes.titlesize"] = 16  # 图表标题字体大小
plt.rcParams["axes.labelsize"] = 14  # 坐标轴标签字体大小
plt.rcParams["xtick.labelsize"] = 12  # x轴刻度字体大小
plt.rcParams["ytick.labelsize"] = 12  # y轴刻度字体大小
plt.rcParams["legend.fontsize"] = 14  # 图例字体大小

# 创建结果文件夹
os.makedirs('./泛化误差实验/theoretical_results', exist_ok=True)

def calculate_theoretical_error():
    # 训练集的范围
    train_range = [0.5, 1.5]
    
    # 读取所有测试集文件
    test_files = ["test_dataset_center_1.5_1.csv"]
    
    if not test_files:
        print('没有找到测试集文件，请先运行datasets.py生成测试集。')
        return
    
    for test_file in test_files:
        print(f'处理测试集: {test_file}')
        # 读取测试集
        test_df = pd.read_csv(f"./datasets/{test_file}")
        num_samples = len(test_df)
        
        # 识别与训练集不重叠的部分
        non_overlap_mask = ~((test_df['x1'] >= train_range[0]) & (test_df['x1'] <= train_range[1]) &
                             (test_df['x2'] >= train_range[0]) & (test_df['x2'] <= train_range[1]))

        

        # 不重叠部分数据
        non_overlap_data = test_df[non_overlap_mask]
        # 新增：重叠部分数据
        overlap_data = test_df[~non_overlap_mask]
        x1 = non_overlap_data['y1'].values
        x2 = non_overlap_data['y2'].values
        x1_range = x1.max() - x1.min()
        
        x2_range = x2.max() - x2.min()
        max_distance = np.sqrt(x1_range**2 + x2_range**2)
        # 计算不重叠部分的概率
        non_overlap_probability = len(non_overlap_data) / len(test_df)
        print(f'不重叠部分的概率: {non_overlap_probability:.4f}')

        # 绘制重叠与不重叠部分数据分布图
        plt.figure(figsize=(12, 8))
        # 绘制重叠部分
        if not overlap_data.empty:
            plt.scatter(overlap_data['x1'], overlap_data['x2'], c='blue', alpha=0.6, label='Overlapping Part')
        # 绘制不重叠部分
        if not non_overlap_data.empty:
            plt.scatter(non_overlap_data['x1'], non_overlap_data['x2'], c='red', alpha=0.6, label='Non-overlapping Part')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Distribution of Overlapping and Non-overlapping Data in {test_file}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 保存图片
        distribution_plot_path = f'./泛化误差实验/theoretical_results/{test_file.replace(".csv", "_distribution.png")}'

        plt.savefig(distribution_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'数据分布图已保存到: {distribution_plot_path}')

        # # 保存结果
        result_file = f'./泛化误差实验/theoretical_results/{test_file.replace(".csv", "_results.txt")}'
        # 移除radial_partition函数和放射状划分逻辑
        
        # 添加x轴和y轴划分函数
        def x_axis_partition(data, n_bins):
            """沿x轴(y1)均匀划分区域"""
            y_vals = data['y1'].values
            bins = np.linspace(y_vals.min(), y_vals.max(), n_bins+1)
            counts, _ = np.histogram(y_vals, bins=bins)
            return counts, bins
        
        def y_axis_partition(data, n_bins):
            """沿y轴(y2)均匀划分区域"""
            y_vals = data['y2'].values
            bins = np.linspace(y_vals.min(), y_vals.max(), n_bins+1)
            counts, _ = np.histogram(y_vals, bins=bins)
            return counts, bins
        
        # 分别计算x轴和y轴的最优网格数量
        optimal_n_x = 1
        for n_bins in range(2, 1000):
            counts, _ = x_axis_partition(non_overlap_data, n_bins)
            if np.any(counts == 0):
                optimal_n_x = n_bins - 1
                break
            optimal_n_x = n_bins
        
        optimal_n_y = 1
        for n_bins in range(2, 1000):
            counts, _ = y_axis_partition(non_overlap_data, n_bins)
            if np.any(counts == 0):
                optimal_n_y = n_bins - 1
                break
            optimal_n_y = n_bins
        
        # 计算x轴划分的熵值
        counts_x, _ = x_axis_partition(non_overlap_data, optimal_n_x)
        prob_x = counts_x / num_samples
        prob_x = prob_x[prob_x > 0]
        discrete_entropy_x = -np.sum(prob_x * np.log(prob_x))
        
        # 计算y轴划分的熵值
        counts_y, _ = y_axis_partition(non_overlap_data, optimal_n_y)
        prob_y = counts_y / num_samples
        prob_y = prob_y[prob_y > 0]
        discrete_entropy_y = -np.sum(prob_y * np.log(prob_y))
        
        # 计算平均熵值作为最终熵值
        final_discrete_entropy = (discrete_entropy_x + discrete_entropy_y) / 2
        
        # 使用平均熵值计算理论误差
        loss_1 = 0.5 * (non_overlap_probability * np.log(num_samples) - final_discrete_entropy)
        loss_val = np.sqrt(loss_1) if loss_1 >= 0 else np.nan
        
        # 更新结果保存
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f'训练集范围: {train_range}\n')
            f.write(f'不重叠部分的概率: {non_overlap_probability:.4f}\n')
            f.write(f'非重叠部分样本数: {non_overlap_data.shape[0]}\n')
            f.write(f'x轴最优网格数量: {optimal_n_x}\n')
            f.write(f'y轴最优网格数量: {optimal_n_y}\n')
            f.write(f'x轴离散熵: {discrete_entropy_x:.4f} bits\n')
            f.write(f'y轴离散熵: {discrete_entropy_y:.4f} bits\n')
            f.write(f'平均离散熵: {final_discrete_entropy:.4f} bits\n')
            f.write(f'理论最大误差值: {loss_val:.4f}\n')
            #f.write(f'测试集最远两点距离: {max_distance:.4f}\n')  # 添加最远点距离信息
        print(f'理论误差结果已保存到: {result_file}')

        #保守估计误差
        # 为每个样本单独随机取点计算误差
        all_distances = []
        for idx, row in non_overlap_data.iterrows():
            # 固定当前样本
            current_y1, current_y2 = row['y1'], row['y2']
            # 从非重叠部分随机取一个点
            random_point = non_overlap_data.sample(n=1, random_state=None)
            random_y1_sample, random_y2_sample = random_point['y1'].values[0], random_point['y2'].values[0]
            # 计算欧氏距离
            distance = np.sqrt((current_y1 - random_y1_sample)**2 + (current_y2 - random_y2_sample)**2)
            all_distances.append(distance)


        non_overlap_data = non_overlap_data.copy()
        non_overlap_data['distance'] = all_distances

        # 重置索引
        non_overlap_data = non_overlap_data.reset_index(drop=True)

        # 模型预测误差计算
        training_steps = [10,30,50,70,100,200,600,1000,4000,7000]
        model_predictions = {}
        conservative_estimates = {}
        
        # 随机选择10个非重叠区测试样本
        np.random.seed(42) 
        if len(non_overlap_data) >= 10:
            selected_indices = np.random.choice(non_overlap_data.index, size=10, replace=False)
            selected_samples = non_overlap_data.iloc[selected_indices].copy()
        else:
            selected_samples = non_overlap_data.copy()
            print(f"警告：非重叠区样本不足10个，使用全部{len(non_overlap_data)}个样本")
        
        print(f"\n开始模型预测误差计算...")
        
        # 计算保守估计误差
        conservative_rmse = []
        for idx, row in selected_samples.iterrows():
            # 从剩余非重叠数据中随机取一个点作为保守估计
            remaining_data = non_overlap_data.drop(idx)
            if len(remaining_data) > 0:
                random_point = remaining_data.sample(n=1, random_state=42)
                conservative_values = random_point[['y1', 'y2']].values[0]
                true_value = row[['y1', 'y2']].values
                conservative_rmse.append(np.sqrt(np.mean((conservative_values - true_value)**2)))
            else:
                conservative_rmse.append(np.nan)
        conservative_estimates = np.array(conservative_rmse)
        
        for step in training_steps:
            checkpoint_path = f'./泛化误差实验/endpoint/model_step_{step}_checkpoint.pth'
            if not os.path.exists(checkpoint_path):
                print(f"警告：找不到检查点文件 {checkpoint_path}")
                continue
                
            # 加载模型
            model = FeedForwardNN()
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            model.eval()
            
            # 预测
            test_input = selected_samples[['x1', 'x2']].values
            test_tensor = torch.FloatTensor(test_input)
            predictions = model.predict(test_tensor)
            
            # 计算均方根误差
            true_values = selected_samples[['y1', 'y2']].values
            rmse_values = np.sqrt(np.mean((predictions - true_values)**2, axis=1))
            model_predictions[step] = rmse_values
            
            print(f"步数 {step}: 平均预测误差 = {np.mean(rmse_values):.4f}")

        # 绘制新的综合图表
        plt.figure(figsize=(15, 10))
        
        # 绘制模型预测误差线
        colors = plt.cm.tab10(np.linspace(0, 1, len(training_steps)))
        for i, (step, rmse_values) in enumerate(model_predictions.items()):
            plt.plot(range(len(rmse_values)), rmse_values, 'o-', color=colors[i], 
                     linewidth=2, markersize=6, label=f'Model (Step {step})')
        
        # 绘制单一保守估计误差线
        plt.plot(range(len(conservative_estimates)), conservative_estimates, 'b--', 
                 linewidth=3, alpha=0.8, label='Conservative Estimation')
        
        # 添加理论误差横线
        theoretical_colors = ['r', 'g', 'orange', 'purple', 'brown']
        if not np.isnan(loss_val):
            plt.axhline(y=loss_val, color=theoretical_colors[0], 
                       linestyle=':', linewidth=3, alpha=0.8,
                       label=f'TVD {loss_val:.4f}')
        
        plt.xlabel('Test Sample Index')
        plt.ylabel('Root Mean Squared Error')
        plt.title('Comparison of Model Prediction Errors, Conservative Estimation Errors,\n and TVD under Different Training Steps in Non-overlapping Regions')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存新图表
        new_plot_path = f'./泛化误差实验/theoretical_results/{test_file.replace(".csv", "_model_comparison.png")}'
        plt.savefig(new_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'模型预测对比图已保存到: {new_plot_path}')

        # 绘制样本到随机点的距离
        plt.figure(figsize=(12, 8))
        
        # 绘制样本到随机点的距离
        plt.plot(non_overlap_data.index, non_overlap_data['distance'], 'b-', linewidth=1, label='Conservative Prediction')
        
        # # 计算并绘制保守预测平均值横线
        # conservative_mean = np.nanmean(conservative_estimates)
        # plt.axhline(y=conservative_mean, color='black', linestyle='-', linewidth=2, alpha=0.9, label=f'Conservative Prediction Average ({conservative_mean:.4f})')
        
        # 为每个网格大小添加理论损失横线
        if not np.isnan(loss_val):
            plt.axhline(y=loss_val, color='r', linestyle='--', linewidth=2,
                       label=f'TVD {loss_val:.4f}')

        plt.xlabel('Number of Test Samples in Non-overlapping Regions')
        plt.ylabel('Root Mean Squared Error')
        plt.title('Comparison of TVD and Conservative Estimation Prediction Errors in Non-overlapping Regions')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        # 保存图表
        plot_path = f'./泛化误差实验/theoretical_results/{test_file.replace(".csv", "_rmse_plot.png")}'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'保守预测对比图已保存到: {plot_path}')

        # -------------------- 新增y轴风格图表 --------------------
        n_y_range = range(1, optimal_n_y+2)
        y_entropy_values = []
        for n in n_y_range:
            counts_temp, _ = y_axis_partition(non_overlap_data, n)
            if np.any(counts_temp == 0):
                y_entropy_values.append(np.nan)
                continue
            prob_temp = counts_temp / num_samples
            prob_temp = prob_temp[prob_temp > 0]
            entropy = -np.sum(prob_temp * np.log(prob_temp))
            y_entropy_values.append(entropy)

        # 绘制y轴熵值变化曲线图
        plt.figure(figsize=(12, 8))
        plt.plot(n_y_range, y_entropy_values, 'go-', linewidth=2, markersize=4)
        plt.xlabel('Number of Y-axis Grids')
        plt.ylabel('Entropy')
        plt.title('Change of Y-axis Partition Entropy with Grid Number')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=optimal_n_y, color='red', linestyle='--', linewidth=2, label=f'Optimal Grid Number ({optimal_n_y})')
        if not np.isnan(y_entropy_values[-1]):
            plt.text(optimal_n_y + 1, y_entropy_values[-1], f'{y_entropy_values[-1]:.4f}', ha='left', va='center', color='r')
        plt.legend()
        y_radial_entropy_path = f'./泛化误差实验/theoretical_results/{test_file.replace(".csv", "_y_axis_radial_entropy.png")}'
        plt.savefig(y_radial_entropy_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'y-axis style entropy plot saved to: {y_radial_entropy_path}')

        # 创建y轴划分的上下子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), gridspec_kw={'height_ratios': [1, 1.5]})
        # 上方子图：y轴划分数据分布
        ax1.scatter(non_overlap_data['y1'], non_overlap_data['y2'], alpha=0.5, color='gray', label='Data Points')
        y_bins = np.linspace(non_overlap_data['y2'].min(), non_overlap_data['y2'].max(), optimal_n_y+1)
        for bin_val in y_bins[1:-1]:
            ax1.axhline(y=bin_val, color='g', linestyle='-', linewidth=1, alpha=0.7)
        ax1.set_xlabel('y1')
        ax1.set_ylabel('y2')
        ax1.set_title(f'Y-axis Partition Data Distribution ({optimal_n_y} Regions)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 下方子图：y轴区域样本数量条形图
        y_labels = [f'{y_bins[i]:.2f}-{y_bins[i+1]:.2f}' for i in range(optimal_n_y)]
        bars = ax2.bar(range(optimal_n_y), counts_y, color=plt.cm.viridis(counts_y / max(counts_y)))
        ax2.set_xlabel('Y-axis Regions')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Distribution of Sample Counts in Y-axis Partitioned Regions')
        ax2.set_xticks(range(optimal_n_y))
        ax2.set_xticklabels(y_labels, rotation=45)
        ax2.tick_params(axis='x', labelsize=4)
        for i, (bar, count) in enumerate(zip(bars, counts_y)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_y)*0.01, str(count), ha='center', va='bottom')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        y_radial_heatmap_path = f'./泛化误差实验/theoretical_results/{test_file.replace(".csv", "_y_axis_radial_heatmap.png")}'
        plt.savefig(y_radial_heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'y轴热力图已保存到: {y_radial_heatmap_path}')
        plt.close()

        # -------------------- 新增x轴风格图表 --------------------
        n_x_range = range(1, optimal_n_x+2)
        x_entropy_values = []
        for n in n_x_range:
            counts_temp, _ = x_axis_partition(non_overlap_data, n)
            if np.any(counts_temp == 0):
                x_entropy_values.append(np.nan)
                continue
            prob_temp = counts_temp / num_samples
            prob_temp = prob_temp[prob_temp > 0]
            entropy = -np.sum(prob_temp * np.log(prob_temp))
            x_entropy_values.append(entropy)

        # 绘制x轴熵值变化曲线图
        plt.figure(figsize=(12, 8))
        plt.plot(n_x_range, x_entropy_values, 'bo-', linewidth=2, markersize=4)
        plt.xlabel('x轴网格数量')
        plt.ylabel('熵值')
        plt.title('x轴划分熵值随网格数量的变化')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=optimal_n_x, color='red', linestyle='--', linewidth=2, label=f'最优网格数 ({optimal_n_x})')
        if not np.isnan(x_entropy_values[-1]):
            plt.text(optimal_n_x + 1, x_entropy_values[-1], f'{x_entropy_values[-1]:.4f}', ha='left', va='center', color='r')
        plt.legend()
        x_radial_entropy_path = f'./泛化误差实验/theoretical_results/{test_file.replace(".csv", "_x_axis_radial_entropy.png")}'
        plt.savefig(x_radial_entropy_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'x轴风格熵值图已保存到: {x_radial_entropy_path}')

        # 创建x轴划分的上下子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), gridspec_kw={'height_ratios': [1, 1.5]})
       # 上方子图：x轴划分数据分布
        ax1.scatter(non_overlap_data['y1'], non_overlap_data['y2'], alpha=0.5, color='gray', label='数据点')
        x_bins = np.linspace(non_overlap_data['y1'].min(), non_overlap_data['y1'].max(), optimal_n_x+1)
        for bin_val in x_bins[1:-1]:
            ax1.axvline(x=bin_val, color='b', linestyle='-', linewidth=1, alpha=0.7)
        ax1.set_xlabel('y1')
        ax1.set_ylabel('y2')
        ax1.set_title(f'x轴划分数据分布（{optimal_n_x}个区域）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

       # 下方子图：x轴区域样本数量条形图
        x_labels = [f'{x_bins[i]:.2f}-{x_bins[i+1]:.2f}' for i in range(optimal_n_x)]
        bars = ax2.bar(range(optimal_n_x), counts_x, color=plt.cm.viridis(counts_x / max(counts_x)))
        ax2.set_xlabel('x轴区域')
        ax2.set_ylabel('样本数量')
        ax2.set_title(f'x轴划分区域样本数量分布')
        ax2.set_xticks(range(optimal_n_x))
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
        ax2.tick_params(axis='x', labelsize=4)
        for i, (bar, count) in enumerate(zip(bars, counts_x)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_x)*0.01, str(count), ha='center', va='bottom')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        x_radial_heatmap_path = f'./泛化误差实验/theoretical_results/{test_file.replace(".csv", "_x_axis_radial_heatmap.png")}'
        plt.savefig(x_radial_heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'x轴热力图已保存到: {x_radial_heatmap_path}')
        plt.close()
if __name__ == '__main__':
    calculate_theoretical_error()
   