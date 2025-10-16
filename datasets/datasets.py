import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 修改字体设置
plt.rcParams["font.family"] = "Arial"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

np.random.seed(43)

def generate_datasets():
    # 数据集参数
    train_samples = 10000  # 训练集样本数量
    test_samples = 10000   # 每个测试集样本数量
    # 四个测试集的中心点
    test_centers = [(1.5, 1)]
    # 圆形分布半径
    circle_radius = 0.5

    # ---------------------- 生成训练集 ----------------------
    # 训练集：x1, x2 在指定范围内均匀分布
    x1_train = np.random.uniform(0.5, 1.5, train_samples)
    x2_train = np.random.uniform(0.5, 1.5, train_samples)

    # 计算y1和y2
    y1_train = np.sqrt(4 * x1_train**2 + 3 * x2_train**2)
    y2_train = np.sqrt(2 * x1_train**2 + x2_train**2)

    # 创建训练集DataFrame
    train_df = pd.DataFrame({
        'x1': x1_train,
        'x2': x2_train,
        'y1': y1_train,
        'y2': y2_train
    })

    # ---------------------- 保存训练集 ----------------------
    train_df.to_csv('./datasets/train_dataset.csv', index=False)
    print('训练集已保存到 ./datasets/train_dataset.csv')
    print(f'训练集大小: {len(train_df)}')

    # ---------------------- 生成测试集 ----------------------
    test_dfs = []
    for i, (center_x, center_y) in enumerate(test_centers, 1):
        # 生成圆形均匀分布数据
        theta = np.random.uniform(0, 2*np.pi, test_samples)  # 角度均匀分布
        r = circle_radius * np.sqrt(np.random.uniform(0, 1, test_samples))  # 半径按平方根分布以保证均匀性
        x1_test = center_x + r * np.cos(theta)
        x2_test = center_y + r * np.sin(theta)

        # 计算y1和y2
        y1_test = np.sqrt(4 * x1_test**2 + 3 * x2_test**2)
        y2_test = np.sqrt(2 * x1_test**2 + x2_test**2)

        # 创建测试集DataFrame
        test_df = pd.DataFrame({
            'x1': x1_test,
            'x2': x2_test,
            'y1': y1_test,
            'y2': y2_test
        })

        # 保存测试集
        filename = f'./datasets/test_dataset_center_{center_x}_{center_y}.csv'
        test_df.to_csv(filename, index=False)
        test_dfs.append(test_df)
        print(f'测试集已保存到 {filename}')
        print(f'测试集大小: {len(test_df)}')

    # ---------------------- 绘制数据分布图 ----------------------
    plt.figure(figsize=(12, 8))

    # 绘制训练集
    #plt.scatter(train_df['x1'], train_df['x2'], alpha=0.5, label='Training Datasets (Uniform Distribution)', s=10, color='slateblue')
    plt.scatter(train_df['x1'], train_df['x2'], alpha=0.5, label='Training Datasets (Uniform Distribution)', s=10, color='blue')

    # 绘制四个测试集
    #colors = ['g', 'green', 'blue', 'purple']
    colors = ['red', 'green', 'blue', 'purple']

    for i, (test_df, (center_x, center_y)) in enumerate(zip(test_dfs, test_centers)):
        plt.scatter(test_df['x1'], test_df['x2'], alpha=0.3, label=f'Test Datasets (Center: ({center_x},{center_y}))(Uniform Distribution)', s=10, color=colors[i])

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Data Distribution')
    plt.legend()
    plt.grid(True)

    # 保存图表
    plt.savefig('./datasets/data_distribution.png', dpi=300)
    plt.show()
    plt.close()
    print('Data distribution plot saved to ./datasets/data_distribution.png')

    return train_df, test_dfs

if __name__ == '__main__':
    generate_datasets()