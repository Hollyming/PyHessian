# Hessian特征向量分析实验

这个实验旨在分析深度神经网络在微调过程中Hessian特征向量的演化特性。我们通过在CIFAR10-C数据集上微调预训练的ResNet20模型，计算不同训练批次的Hessian矩阵特征值和特征向量，研究参数空间曲率的变化情况及特征向量子空间的一致性。

## 实验原理

Hessian矩阵是损失函数相对于模型参数的二阶导数矩阵，它描述了参数空间中的曲率信息。Hessian矩阵的特征值和特征向量可以揭示优化过程中参数空间的几何特性：

1. **特征值**：表示不同方向上的曲率大小
   - 大的正特征值表示该方向上的强凸性
   - 小的特征值表示平坦区域
   - 负特征值表示鞍点方向

2. **特征向量**：表示参数空间中的主要变化方向
   - 对应大特征值的特征向量表示高曲率方向
   - 这些方向在训练过程中可能相对稳定或发生变化

本实验的核心思想是：如果在微调过程中，Hessian的主要特征向量维持在一个相对稳定的子空间中，那么这表明模型的优化过程具有一定的方向一致性。通过对不同批次收集的特征向量进行PCA分析，我们可以观察这种一致性的存在与否。

## 实验内容

1. 从预训练的ResNet20模型出发，在CIFAR10-C数据集(高斯噪声level-5)上进行微调
2. 在微调过程中，每隔一定批次计算Hessian矩阵的前r个最大特征值及对应特征向量
3. 将所有批次的特征向量组织成一个大矩阵U
4. 对矩阵U进行PCA分析，观察特征向量子空间的结构
5. 分析不同批次间特征向量的关联性

## 文件说明

- `finetune_and_analyze_hessian.py`: 主要实验脚本
- `run_finetune_analysis.sh`: 用于执行实验的Bash脚本
- `README_hessian_analysis.md`: 本说明文档

## 使用方法

### 准备工作

1. 确保已安装PyTorch, NumPy, Matplotlib, scikit-learn等依赖包
   ```bash
   pip install torch torchvision numpy matplotlib scikit-learn tqdm requests
   ```

2. 关于CIFAR10-C数据集：
   - **自动下载**：脚本会自动下载CIFAR10-C数据集(约1.1GB)，无需手动准备
   - **手动下载**：如果您希望手动下载，可以从 [CIFAR10-C官方链接](https://zenodo.org/record/2535967) 获取并解压到`--cifar10c-path`指定的目录

3. 确保预训练的ResNet20模型在`checkpoints/resnet20_cifar10.pkl`或通过`--resume`参数指定的路径可用

### 运行实验

使用默认参数运行实验：

```bash
chmod +x run_finetune_analysis.sh
./run_finetune_analysis.sh
```

自定义参数：

```bash
./run_finetune_analysis.sh --epochs 5 --lr 0.0005 --save-interval 5 --r 15 --noise-type gaussian_noise --corruption-level 5
```

主要参数说明：

- `--batch-size`: 微调的批量大小
- `--hessian-batch-size`: 计算Hessian的批量大小
- `--epochs`: 微调的轮数
- `--lr`: 学习率
- `--r`: 计算前r个特征值/特征向量
- `--save-interval`: 保存Hessian的batch间隔
- `--resume`: 预训练模型路径
- `--cifar10c-path`: CIFAR10-C数据集路径
- `--noise-type`: CIFAR10-C噪声类型（可选值见下文）
- `--corruption-level`: CIFAR10-C噪声级别(1-5)
- `--save-dir`: 结果保存目录

### CIFAR10-C噪声类型

CIFAR10-C包含15种不同类型的噪声/失真，每种有5个强度级别：

- `gaussian_noise`: 高斯噪声
- `shot_noise`: 散粒噪声
- `impulse_noise`: 脉冲噪声
- `defocus_blur`: 失焦模糊
- `glass_blur`: 玻璃模糊
- `motion_blur`: 运动模糊
- `zoom_blur`: 缩放模糊
- `snow`: 雪花效果
- `frost`: 霜冻效果
- `fog`: 雾效果
- `brightness`: 亮度变化
- `contrast`: 对比度变化
- `elastic_transform`: 弹性变形
- `pixelate`: 像素化
- `jpeg_compression`: JPEG压缩失真

### 输出结果

实验结果将保存在`--save-dir`指定的目录中，包括：

1. **hessian_matrices/**: 保存的Hessian矩阵
2. **eigen_vectors/**: 每个批次的特征值和特征向量
3. **models/**: 保存的模型检查点
4. **pca_results/**: PCA分析结果
5. **plots/**: 可视化图表
   - `pca_variance.png`: PCA解释方差比例图
   - `pca_cumulative_variance.png`: PCA累积解释方差图
   - `pca_projection.png`: 特征向量在前两个主成分上的投影
   - `cosine_similarity.png`: 不同批次特征向量间的余弦相似度热图
6. **hessian_analysis.log**: 实验日志

## 实验结果分析

实验输出的几个重要可视化结果可以帮助我们理解Hessian特征向量的演化特性：

1. **PCA解释方差比例**：如果前几个主成分能解释大部分方差，说明特征向量空间具有低维结构
2. **特征向量在主成分上的投影**：如果不同批次的特征向量在PCA空间中聚集成组，表明存在批次相关的子空间结构
3. **余弦相似度热图**：显示不同批次特征向量之间的相似性，可以观察时间上的连续性和突变

通过这些分析，我们可以得出关于神经网络优化过程中参数空间几何特性的洞见。

## 参考文献

1. Yao, Z., Gholami, A., et al. "PyHessian: Neural Networks Through the Lens of the Hessian"
2. Sagun, L., Evci, U., et al. "Empirical Analysis of the Hessian of Over-Parametrized Neural Networks"
3. Ghorbani, B., Krishnan, S., et al. "Investigation of Neural Net Optimization by Random Matrix Theory"
4. Hendrycks, D., Dietterich, T. "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations" 