# Machine Learning Theory Meta-Framework (MLT-MF)

English version of the paper:  
- [Information Science Principles of Machine Learning: A Causal Chain Meta-Framework Based on Formalized Information Mapping](https://arxiv.org/abs/2505.13182)

Chinese version of the paper: 
- [机器学习的信息科学原理：基于形式化信息映射的因果链元框架](https://chinaxiv.org/abs/202505.00162v6?locale=zh_CN)

---

## 📖 背景

当前机器学习理论研究存在以下问题：
- 缺乏覆盖全生命周期的统一形式化框架；
- 信息流的因果链缺乏明确表达；
- 可解释性与伦理安全缺乏严谨的数学基础。

为解决这些挑战，本文基于**客观信息理论（Objective Information Theory, OIT）**，提出了**机器学习理论元框架（MLT-MF）**，通过形式化信息映射统一建模机器学习全过程，建立可解释性与伦理安全的普适定义，并给出模型泛化误差（TVD 上界）的理论估计。

---

## 🏗️ 核心贡献

1. **统一框架**：提出 MLT-MF，将初始化、训练、输入、输出、优化等阶段纳入形式逻辑体系。
2. **可解释性**：首次基于形式逻辑系统给出模型可解释性的严格定义与定理。
3. **伦理安全**：提出可验证的伦理安全形式化定义与保证方法，基于超图算法构造最大伦理安全输出。
4. **泛化误差**：给出总变差距离（TVD）的上界估计，解释了“问题空间必须包含在模型知识空间”这一核心原理。
5. **实验验证**：通过简单前馈神经网络模型，验证了 MLT-MF 的解释性、伦理安全性与泛化误差上界估计。

---

## 📂 仓库结构  
  
├── datasets/ # 论文实验所用数据集  
├── Generalization_error_experiment/ # 泛化误差实验代码  
├── Interpretability_experiments/ # 可解释性实验代码  
├── Ethical_experiments/ # 伦理检测实验代码  
└── README.md # 本说明文档  
  

## 🚀 使用方法

### 1. 环境准备
```bash
git clone https://github.com/xuhu6736/MLT-MF.git
cd MLT-MF
