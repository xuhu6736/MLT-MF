import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 用于设置中文字体
import numpy as np
import sys
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义结果文件路径
results_file_path = r'补充实验2\tvd_results.txt'

# 打开结果文件用于写入
with open(results_file_path, 'w', encoding='utf-8') as results_file:
    # 重定向标准输出到文件和控制台
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()  # 确保立即写入
        def flush(self):
            for f in self.files:
                f.flush()
    
    # 创建标准输出的备份
    original_stdout = sys.stdout
    
    # 重定向标准输出
    sys.stdout = Tee(sys.stdout, results_file)
    
    try:
        # 开始计算和输出
        dataset_path = r'补充实验2\new_test_dataset.csv'
        test_df = pd.read_csv(dataset_path)

        # oq-
        grid0_probability_sum = test_df[test_df['grid0_flag'] == 0]['probability'].sum()
        # 留下的测试集
        grid_1_probability_sum = test_df[test_df['grid0_flag'] == -1]['probability'].sum()
        # 计算所有数据的概率和
        total_probability_sum = test_df['probability'].sum()

        print(f"丢失质量: {grid0_probability_sum:.4f}")
        # print(f"标签为 1 的数据的概率和为: {grid_1_probability_sum:.4f}")
        # print(f"所有数据的概率和为: {total_probability_sum:.4f}")

        # 读取test1_new_dataset.csv
        test1_dataset_path = r'补充实验2\test1_new_dataset.csv'
        # oq+
        test1_df = pd.read_csv(test1_dataset_path)

        # 计算test1数据的所有概率和
        test1_total_probability_sum = test1_df['probability'].sum()

        print(f"叠加质量: {test1_total_probability_sum:.4f}")

        # 创建新的噪声输入信息分布数组
        noise_input_df = test_df[test_df['grid0_flag'] == -1]
        # 合并noise_input_df和test1_df
        combined_df = pd.concat([noise_input_df, test1_df])
        # 计算归一化因子
        normalization_factor = 1 - grid0_probability_sum + test1_total_probability_sum
        # 对每个元素进行归一化
        combined_df['normalized_probability'] = combined_df['probability'] / normalization_factor
        # 计算并集的概率和
        combined_probability_sum = combined_df['normalized_probability'].sum()

        print(f"噪声输入信息分布并集的概率和: {combined_probability_sum:.4f}")

        # 计算未覆盖质量
        # 筛选出x1和x2都不在(0.5, 1.5)范围内的点
        uncovered_df = combined_df[
            (combined_df['x1'] < 0.5) | (combined_df['x1'] > 1.5) |
            (combined_df['x2'] < 0.5) | (combined_df['x2'] > 1.5)
        ].copy()  # 使用.copy()避免SettingWithCopyWarning
        # 计算这些点的概率和
        uncovered_probability_sum = uncovered_df['normalized_probability'].sum()

        print(f"未覆盖质量: {uncovered_probability_sum:.4f}")

        # 计算覆盖区域的点
        covered_df = combined_df[
            (combined_df['x1'] >= 0.5) & (combined_df['x1'] <= 1.5) &
            (combined_df['x2'] >= 0.5) & (combined_df['x2'] <= 1.5)
        ].copy()

        # 计算未覆盖域条件分布
        # 对uncovered_df进行归一化
        uncovered_df['conditional_probability'] = uncovered_df['normalized_probability'] / uncovered_probability_sum

        # 绘制覆盖区域和未覆盖区域的散点图
        plt.figure(figsize=(10, 8))
        # 绘制覆盖区域的点（蓝色）
        if not covered_df.empty:
            plt.scatter(covered_df['y1'], covered_df['y2'], c='blue', alpha=0.6, label='覆盖区域', s=20)
        # 绘制未覆盖区域的点（红色）
        if not uncovered_df.empty:
            plt.scatter(uncovered_df['y1'], uncovered_df['y2'], c='red', alpha=0.6, label='未覆盖区域', s=20)

        # 计算未覆盖区域点的y1和y2的极值及差值
        if not uncovered_df.empty:
            y1_min_uncovered = uncovered_df['y1'].min()
            y1_max_uncovered = uncovered_df['y1'].max()
            y1_diff = y1_max_uncovered - y1_min_uncovered
            
            y2_min_uncovered = uncovered_df['y2'].min()
            y2_max_uncovered = uncovered_df['y2'].max()
            y2_diff = y2_max_uncovered - y2_min_uncovered

            print(f"未覆盖区域点的y1差值: {y1_diff:.4f}")
            
            print(f"未覆盖区域点的y2差值: {y2_diff:.4f}")
        else:
            print("未覆盖区域为空，无法计算极值")

        plt.xlabel('y1')
        plt.ylabel('y2')
        plt.title('覆盖区域与未覆盖区域散点图')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 构建二维正态分布计算式
        if not uncovered_df.empty:
            # 计算均值和方差
            mu1 = y1_diff / 2  # y1的均值
            mu2 = y2_diff / 2  # y2的均值
            sigma1 = (y1_diff / 6)**0.5  # y1的标准差
            sigma2 = (y2_diff / 6)**0.5  # y2的标准差
            
            print(f"\n二维正态分布参数:")
            print(f"均值: μ1 = {mu1:.4f}, μ2 = {mu2:.4f}")
            print(f"方差: σ1² = {sigma1**2:.4f}, σ2² = {sigma2**2:.4f}")
            
            # 定义二维正态分布的概率密度函数
            def bivariate_normal_pdf(y1, y2, mu1, mu2, sigma1, sigma2):
                """计算二维正态分布的概率密度函数值"""
                # 标准化变量
                z1 = (y1 - mu1) / sigma1
                z2 = (y2 - mu2) / sigma2
                
                # 计算概率密度
                exponent = -0.5 * (z1**2 + z2**2)
                coefficient = 1 / (2 * np.pi * sigma1 * sigma2)
                pdf = coefficient * np.exp(exponent)
                
                return pdf
            
            # 添加KL散度计算
            if not uncovered_df.empty:
                # 读取网格信息文件
                new_test_grid_path = r'补充实验2\points_with_grid_info\new_test_points_with_grid_info.csv'
                test1_grid_path = r'补充实验2\points_with_grid_info\test1_points_with_grid_info.csv'
                
                # 读取网格数据
                new_test_grid_df = pd.read_csv(new_test_grid_path)
                test1_grid_df = pd.read_csv(test1_grid_path)
                
                # 合并网格数据
                combined_grid_df = pd.concat([new_test_grid_df, test1_grid_df])
                
                # 筛选出未覆盖区域的网格信息（根据x1,x2范围）
                uncovered_grid_df = combined_grid_df[
                    (combined_grid_df['x1'] < 0.5) | (combined_grid_df['x1'] > 1.5) |
                    (combined_grid_df['x2'] < 0.5) | (combined_grid_df['x2'] > 1.5)
                ].copy()
                
                # 定义二维正态分布的积分函数
                def integrate_bivariate_normal_over_grid(y1_start, y1_end, y2_start, y2_end, mu1, mu2, sigma1, sigma2):
                    from scipy.integrate import dblquad
                    
                    def integrand(y2, y1):
                        return bivariate_normal_pdf(y1, y2, mu1, mu2, sigma1, sigma2)
                    
                    integral, error = dblquad(integrand, y1_start, y1_end, 
                                            lambda y1: y2_start, lambda y1: y2_end)
                    
                    return integral
                
                # 计算KL散度
                kl_divergences = []
                
                # 为每个未覆盖区域的点计算KL散度
                for idx, point in uncovered_df.iterrows():
                    y1_val = point['y1']
                    y2_val = point['y2']
                    prob = point['conditional_probability']
                    
                    # 找到对应的网格
                    matching_grid = uncovered_grid_df[
                        (uncovered_grid_df['y1'] >= y1_val - 0.01) & (uncovered_grid_df['y1'] <= y1_val + 0.01) &
                        (uncovered_grid_df['y2'] >= y2_val - 0.01) & (uncovered_grid_df['y2'] <= y2_val + 0.01)
                    ]
                    
                    if not matching_grid.empty:
                        # 取第一个匹配的网格
                        grid_info = matching_grid.iloc[0]
                        
                        # 获取网格边界
                        y1_start = grid_info['grid_y1_start']
                        y1_end = grid_info['grid_y1_end']
                        y2_start = grid_info['grid_y2_start']
                        y2_end = grid_info['grid_y2_end']
                        
                        # 计算该网格的二维正态分布积分
                        q_prob = integrate_bivariate_normal_over_grid(
                            y1_start, y1_end, y2_start, y2_end, mu1, mu2, sigma1, sigma2
                        )
                        
                        # 计算KL散度项
                        if q_prob > 0 and prob > 0:
                            kl_term = prob * np.log(prob / q_prob)
                            kl_divergences.append({
                                'y1': y1_val,
                                'y2': y2_val,
                                'conditional_probability': prob,
                                'q_probability': q_prob,
                                'kl_divergence': kl_term
                            })
                
                # 创建KL散度数据框
                kl_df = pd.DataFrame(kl_divergences)
                
                # 计算总KL散度
                if not kl_df.empty:
                    total_kl_divergence = kl_df['kl_divergence'].sum()
                    print(f"总KL散度: {total_kl_divergence:.6f}")
                    
                    
                else:
                    print("未找到匹配的网格信息来计算KL散度")

                info517 = (1/2 * (total_kl_divergence * uncovered_probability_sum))**0.5
                print(f"5.17公式右侧上限: {info517:.6f}")

                info5181 = grid0_probability_sum + info517
                print(f"5.18公式右侧上限1: {info5181:.6f}")

                info5182 = test1_total_probability_sum/(1 - grid0_probability_sum + test1_total_probability_sum) + info517
                print(f"5.18公式右侧上限2: {info5182:.6f}")
    
    finally:
        # 恢复标准输出
        sys.stdout = original_stdout
        print(f"所有输出已保存到: {results_file_path}")
    

