#!/usr/bin/env python
# -*- coding: utf-8 -*-

#*
# @file 在CIFAR10-C上微调模型并分析Hessian特征值与特征向量
# Copyright (c) Based on PyHessian library.
#*

from __future__ import print_function

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange#显示进度条
from sklearn.decomposition import PCA
import requests
import tarfile
import gzip
import shutil
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter  # 添加 TensorBoard 导入

# 导入本地模块
from utils import *
from models.resnet import resnet
from pyhessian import hessian
from density_plot import get_esd_plot, density_generate

import argparse

# 参数设置
parser = argparse.ArgumentParser(description='在CIFAR10-C上微调并分析Hessian')
parser.add_argument('--batch-size', type=int, default=128, help='微调的批量大小 (default: 128)')
parser.add_argument('--hessian-batch-size', type=int, default=128, help='计算Hessian的批量大小 (default: 128)')
parser.add_argument('--mini-hessian-batch-size', type=int, default=128, help='计算Hessian的mini批量大小 (default: 128)')
parser.add_argument('--epochs', type=int, default=2, help='微调的epoch数 (default: 2)')
parser.add_argument('--lr', type=float, default=0.001, help='学习率 (default: 0.001)')
parser.add_argument('--r', type=int, default=10, help='计算前r个特征值/特征向量 (default: 10)')
parser.add_argument('--save-interval', type=int, default=10, help='保存Hessian的batch间隔 (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, help='记录训练信息的batch间隔 (default: 10)')
parser.add_argument('--resume', type=str, default='./checkpoints/resnet20_cifar10.pkl', help='预训练模型路径')
parser.add_argument('--cifar10c-path', type=str, default='./data/CIFAR-10-C', help='CIFAR10-C数据集路径')
parser.add_argument('--noise-type', type=str, default='gaussian_noise', help='CIFAR10-C噪声类型')
parser.add_argument('--corruption-level', type=int, default=5, help='CIFAR10-C噪声级别 (1-5)')
parser.add_argument('--seed', type=int, default=1, help='随机种子 (default: 1)')
parser.add_argument('--cuda', action='store_false', help='是否使用CUDA')
parser.add_argument('--save-dir', type=str, default='./hessian_analysis', help='结果保存目录')
parser.add_argument('--tensorboard-dir', type=str, default='./runs', help='TensorBoard日志目录 (default: ./runs)')

args = parser.parse_args()

# 设置随机种子
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 创建保存目录
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if not os.path.exists(os.path.join(args.save_dir, 'hessian_matrices')):
    os.makedirs(os.path.join(args.save_dir, 'hessian_matrices'))
if not os.path.exists(os.path.join(args.save_dir, 'eigen_vectors')):
    os.makedirs(os.path.join(args.save_dir, 'eigen_vectors'))
if not os.path.exists(os.path.join(args.save_dir, 'models')):
    os.makedirs(os.path.join(args.save_dir, 'models'))
if not os.path.exists(os.path.join(args.save_dir, 'pca_results')):
    os.makedirs(os.path.join(args.save_dir, 'pca_results'))
if not os.path.exists(os.path.join(args.save_dir, 'plots')):
    os.makedirs(os.path.join(args.save_dir, 'plots'))
if not os.path.exists(args.tensorboard_dir):
    os.makedirs(args.tensorboard_dir)

# 设置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(args.save_dir, f'hessian_analysis.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

#下载CIFAR10-C数据集
def download_cifar10c(root, noise_type=None):
    """下载CIFAR10-C数据集"""
    os.makedirs(root, exist_ok=True)
    cifar10c_url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    tar_file_path = os.path.join(root, "CIFAR-10-C.tar")
    
    # 检查数据文件是否直接存在于指定目录
    if os.path.exists(os.path.join(root, "labels.npy")) and \
       (noise_type is None or os.path.exists(os.path.join(root, f"{noise_type}.npy"))):
        logger.info(f"CIFAR10-C数据集已存在于 {root}，跳过下载")
        return True
        
    # 检查数据文件是否存在于嵌套目录
    nested_dir = os.path.join(root, "CIFAR-10-C")
    if os.path.exists(os.path.join(nested_dir, "labels.npy")) and \
       (noise_type is None or os.path.exists(os.path.join(nested_dir, f"{noise_type}.npy"))):
        logger.info(f"CIFAR10-C数据集已存在于嵌套目录 {nested_dir}，将复制到目标目录")
        try:
            # 列出并移动所有文件到上层目录
            nested_files = os.listdir(nested_dir)
            for file_name in nested_files:
                src_path = os.path.join(nested_dir, file_name)
                dst_path = os.path.join(root, file_name)
                if not os.path.exists(dst_path):
                    logger.info(f"复制文件 {file_name} 到目标目录")
                    shutil.copy2(src_path, dst_path)
            logger.info(f"文件复制完成")
            return True
        except Exception as e:
            logger.error(f"复制文件时出错: {str(e)}")
            # 继续下载流程
    
    # 下载数据集
    if not os.path.exists(tar_file_path):
        logger.info(f"开始下载CIFAR10-C数据集...")
        
        try:
            # 使用Stream模式下载大文件，显示进度
            response = requests.get(cifar10c_url, stream=True)
            response.raise_for_status()  # 检查下载是否成功
            
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            
            with open(tar_file_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                logger.error("下载出错，文件大小不匹配")
                return False
                
        except Exception as e:
            logger.error(f"下载CIFAR10-C数据集时出错: {str(e)}")
            return False
    
    # 解压数据集
    try:
        logger.info(f"解压CIFAR10-C数据集...")
        with tarfile.open(tar_file_path) as tar:
            tar.extractall(path=root)
        logger.info(f"CIFAR10-C数据集解压完成")
        
        # 处理嵌套目录
        if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
            nested_files = os.listdir(nested_dir)
            for file_name in nested_files:
                src_path = os.path.join(nested_dir, file_name)
                dst_path = os.path.join(root, file_name)
                if not os.path.exists(dst_path):
                    logger.info(f"移动文件 {file_name} 到目标目录")
                    shutil.copy2(src_path, dst_path)
            logger.info(f"文件移动完成")
    except Exception as e:
        logger.error(f"解压CIFAR10-C数据集时出错: {str(e)}")
        return False
    
    # 检查必要文件是否存在
    if not os.path.exists(os.path.join(root, "labels.npy")) or \
       (noise_type is not None and not os.path.exists(os.path.join(root, f"{noise_type}.npy"))):
        logger.error(f"下载和解压后仍然找不到必要的数据文件")
        return False
    
    # 可选：清理tar文件
    # os.remove(tar_file_path)
    
    return True

def load_cifar10c(corruption, level, transform=None):
    """加载CIFAR10-C数据集，如果不存在则自动下载"""
    import numpy as np
    
    # 创建数据目录
    if not os.path.exists(args.cifar10c_path):
        os.makedirs(args.cifar10c_path, exist_ok=True)
    
    # 检查数据集是否存在，不存在则下载
    if not os.path.exists(os.path.join(args.cifar10c_path, f"{corruption}.npy")) or \
       not os.path.exists(os.path.join(args.cifar10c_path, "labels.npy")):
        success = download_cifar10c(args.cifar10c_path, corruption)
        if not success:
            raise ValueError(f"无法下载CIFAR10-C数据集")
    
    # 加载图像
    x_path = os.path.join(args.cifar10c_path, f"{corruption}.npy")
    if not os.path.exists(x_path):
        raise ValueError(f"找不到噪声类型: {corruption}.npy")
    
    images = np.load(x_path)
    
    # 加载标签
    y_path = os.path.join(args.cifar10c_path, "labels.npy")
    if not os.path.exists(y_path):
        raise ValueError("找不到标签文件: labels.npy")
    
    labels = np.load(y_path)
    
    # 选择指定级别的数据
    level_idx = (level - 1) * 10000
    images = images[level_idx:level_idx + 10000]
    labels = labels[level_idx:level_idx + 10000]
    
    # 转换为PyTorch数据集
    class CIFAR10C(torch.utils.data.Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            img = self.images[idx]
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
    
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    dataset = CIFAR10C(images, labels, transform)
    return dataset

def compute_hessian(model, criterion, data_loader, device, r=10):
    """计算Hessian矩阵的特征值和特征向量"""
    model.eval()  # 切换到评估模式
    
    # 准备一个batch的数据用于计算Hessian
    hessian_data = []
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        hessian_data.append((inputs, targets))
        #如果从dataloader提取数据集大小大于等于Hessian的batch大小，则停止
        if len(hessian_data) * inputs.size(0) >= args.hessian_batch_size:
            break
    
    # 初始化Hessian计算，pyhessian.hessian()类
    hessian_comp = hessian(model, criterion, dataloader=hessian_data, cuda=(device != 'cpu'))
    
    # 计算特征值和特征向量，pyhessian.hessian()类中的eigenvalues()方法
    eigenvalues, eigenvectors = hessian_comp.eigenvalues(top_n=r)#top_n=r表示计算前r个特征值和特征向量
    
    # 计算Hessian的迹，pyhessian.hessian()类中的trace()方法
    trace = hessian_comp.trace()
    
    # 确保将结果转移到CPU，并转换为标准Python列表
    if isinstance(eigenvalues, torch.Tensor):
        eigenvalues = [val.cpu().item() if isinstance(val, torch.Tensor) else val for val in eigenvalues]
    else:
        eigenvalues = [float(val) for val in eigenvalues]
    
    # 对特征向量进行统一预处理，用于后续统一存储
    processed_eigenvectors = []
    for eigenvector in eigenvectors:
        # 处理每一个特征向量
        processed_vec = []
        for param_tensor in eigenvector:
            if isinstance(param_tensor, torch.Tensor):
                # 将tensor转为numpy并展平
                processed_vec.append(param_tensor.cpu().numpy().flatten())
            else:
                # 确保其他类型也被展平
                processed_vec.append(np.array(param_tensor).flatten())
        
        # 将特征向量中的所有参数连接成一个长向量
        flattened = np.concatenate(processed_vec)#展平为一维长向量
        processed_eigenvectors.append(flattened)#将展平后的特征向量添加到processed_eigenvectors列表中，长度为r    
    return eigenvalues, processed_eigenvectors, trace

def main():
    # 记录开始时间
    start_time = time.time()
    
    # 记录参数
    logger.info(f"实验参数: {vars(args)}")
    
    # 初始化 TensorBoard
    writer = SummaryWriter(os.path.join(args.tensorboard_dir, 
                          f"{args.noise_type}_level{args.corruption_level}_{time.strftime('%Y%m%d-%H%M%S')}"))
    logger.info(f"TensorBoard日志保存在: {writer.log_dir}")
    
    # 设置设备
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载CIFAR10-C数据集
    logger.info(f"加载CIFAR10-C数据集: {args.noise_type}, 级别: {args.corruption_level}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    cifar10c_dataset = load_cifar10c(args.noise_type, args.corruption_level, transform)
    train_loader = torch.utils.data.DataLoader(
        cifar10c_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    logger.info(f"数据集大小: {len(cifar10c_dataset)}")
    
    # 加载验证数据集 (使用原始CIFAR10测试集)
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 加载预训练模型
    logger.info(f"加载预训练模型: {args.resume}")
    model = resnet(num_classes=10, depth=20)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.resume))
    
    # 记录模型结构和参数到TensorBoard
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # CIFAR-10图像大小为32x32，3通道
    writer.add_graph(model, dummy_input)
    
    # 记录模型参数直方图
    for name, param in model.named_parameters():
        writer.add_histogram(f'parameters/{name}', param, 0)  # 记录初始状态
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # 初始化存储特征向量的列表
    all_eigenvectors = []
    all_eigenvalues = []
    batch_indices = []
    
    # 在微调前计算Hessian特征值
    logger.info("在微调前计算Hessian特征值和特征向量")
    eigenvalues, eigenvectors, trace = compute_hessian(
        model, criterion, train_loader, device, r=args.r)
    
    logger.info(f"微调前的Top-{args.r}特征值: {eigenvalues}")
    logger.info(f"微调前的Trace: {np.mean(trace)}")
    
    # 记录初始Hessian特征值到TensorBoard
    for i, eigenvalue in enumerate(eigenvalues):
        writer.add_scalar(f'hessian/eigenvalue_{i+1}', eigenvalue, 0)
    writer.add_scalar('hessian/trace', np.mean(trace), 0)
    
    # 保存微调前的特征值和特征向量
    np.savez(
        os.path.join(args.save_dir, 'eigen_vectors', f'pretrained_eigen.npz'),
        eigenvalues=np.array(eigenvalues),
        eigenvectors=np.array(eigenvectors)  # 现在eigenvectors已经是预处理过的numpy数组
    )
    all_eigenvectors.extend(eigenvectors)
    all_eigenvalues.append(eigenvalues)
    batch_indices.append(0)  # 预训练模型标记为第0个batch
    
    # 微调模型并收集Hessian信息
    logger.info("开始微调模型并收集Hessian信息")
    
    global_batch_idx = 0
    for epoch in range(args.epochs):
        logger.info(f"Epoch: {epoch+1}/{args.epochs}")
        model.train()
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        samples_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            global_batch_idx += 1
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / len(data)
            
            # 记录到TensorBoard
            writer.add_scalar('training/batch_loss', loss.item(), global_batch_idx)
            writer.add_scalar('training/batch_accuracy', accuracy, global_batch_idx)
            
            # 累积epoch统计
            epoch_loss += loss.item() * len(data)
            epoch_acc += correct
            samples_count += len(data)
            
            if batch_idx % args.log_interval == 0:
                logger.info(f"Train Epoch: {epoch+1} [{batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f} Accuracy: {accuracy:.2f}%")
            
            # 按照指定间隔计算Hessian并保存
            if global_batch_idx % args.save_interval == 0:
                logger.info(f"计算Batch {global_batch_idx}的Hessian特征值和特征向量")
                
                # 保存当前模型
                model_path = os.path.join(args.save_dir, 'models', f'model_batch_{global_batch_idx}.pth')
                torch.save(model.state_dict(), model_path)
                logger.info(f"保存模型到: {model_path}")
                
                # 计算Hessian特征值和特征向量
                eigenvalues, eigenvectors, trace = compute_hessian(
                    model, criterion, train_loader, device, r=args.r)
                
                logger.info(f"Batch {global_batch_idx}的Top-{args.r}特征值: {eigenvalues}")
                logger.info(f"Batch {global_batch_idx}的Trace: {np.mean(trace)}")
                
                # 记录Hessian特征值到TensorBoard
                for i, eigenvalue in enumerate(eigenvalues):
                    writer.add_scalar(f'hessian/eigenvalue_{i+1}', eigenvalue, global_batch_idx)
                writer.add_scalar('hessian/trace', np.mean(trace), global_batch_idx)
                
                # 保存特征值和特征向量
                eigen_path = os.path.join(args.save_dir, 'eigen_vectors', f'eigen_batch_{global_batch_idx}.npz')
                np.savez(eigen_path, 
                         eigenvalues=np.array(eigenvalues),
                         eigenvectors=np.array(eigenvectors)  # 现在eigenvectors已经是预处理过的numpy数组
                )
                logger.info(f"保存特征值和特征向量到: {eigen_path}")
                logger.info(f"--------------------------------")
                
                # 收集特征向量用于后续PCA分析
                all_eigenvectors.extend(eigenvectors)
                all_eigenvalues.append(eigenvalues)
                batch_indices.append(global_batch_idx)
        
        # 每个epoch结束时计算并记录平均损失和准确率
        avg_epoch_loss = epoch_loss / samples_count
        avg_epoch_acc = 100. * epoch_acc / samples_count
        writer.add_scalar('training/epoch_loss', avg_epoch_loss, epoch)
        writer.add_scalar('training/epoch_accuracy', avg_epoch_acc, epoch)
        logger.info(f"Epoch {epoch+1} Average - Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_acc:.2f}%")
        
        # 在验证集上评估模型
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item() * len(data)  # 累积批量损失
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_loss /= len(test_loader.dataset)
        val_acc = 100. * correct / len(test_loader.dataset)
        
        # 记录验证集结果到TensorBoard
        writer.add_scalar('validation/loss', val_loss, epoch)
        writer.add_scalar('validation/accuracy', val_acc, epoch)
        logger.info(f"验证集 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # 添加学习率记录
        for param_group in optimizer.param_groups:
            writer.add_scalar('training/learning_rate', param_group['lr'], epoch)
            
        # 记录模型参数直方图
        for name, param in model.named_parameters():
            writer.add_histogram(f'parameters/{name}', param, epoch + 1)  # epoch+1确保与初始状态（0）区分开
    
    # 在全部微调后计算Hessian
    logger.info("在微调后计算Hessian特征值和特征向量")
    eigenvalues, eigenvectors, trace = compute_hessian(
        model, criterion, train_loader, device, r=args.r)
    
    logger.info(f"微调后的Top-{args.r}特征值: {eigenvalues}")
    logger.info(f"微调后的Trace: {np.mean(trace)}")
    
    # 记录最终Hessian特征值到TensorBoard
    for i, eigenvalue in enumerate(eigenvalues):
        writer.add_scalar(f'hessian/eigenvalue_{i+1}', eigenvalue, global_batch_idx + 1)
    writer.add_scalar('hessian/trace', np.mean(trace), global_batch_idx + 1)
    
    # 保存微调后的特征值和特征向量
    np.savez(
        os.path.join(args.save_dir, 'eigen_vectors', f'final_eigen.npz'),
        eigenvalues=np.array(eigenvalues),
        eigenvectors=np.array(eigenvectors)  # 现在eigenvectors已经是预处理过的numpy数组
    )
    all_eigenvectors.extend(eigenvectors)
    all_eigenvalues.append(eigenvalues)
    batch_indices.append(-1)  # 微调后的模型标记为-1
    
    # 保存微调后的模型
    final_model_path = os.path.join(args.save_dir, 'models', 'model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"保存最终模型到: {final_model_path}")
    
    # 对收集的特征向量进行PCA分析
    logger.info("对收集的特征向量进行PCA分析")
    
    # 将特征向量转换为矩阵 - 现在已经是统一格式的numpy数组，可以直接堆叠
    eigenvector_matrix = np.vstack(all_eigenvectors)
    logger.info(f"特征向量矩阵形状: {eigenvector_matrix.shape}")#特征向量矩阵形状为(r*B,d)
    #resnet20的参数量d=272474
    
    # 保存原始特征向量矩阵
    np.save(os.path.join(args.save_dir, 'pca_results', 'eigenvector_matrix.npy'), eigenvector_matrix)
    
    # 应用PCA
    pca = PCA(n_components=20)  # 保留前20个主成分
    pca_result = pca.fit_transform(eigenvector_matrix.T)  # 转置后进行PCA
    
    # 保存PCA结果
    np.savez(
        os.path.join(args.save_dir, 'pca_results', 'pca_result.npz'),
        pca_result=pca_result,
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
        components=pca.components_,
        batch_indices=batch_indices
    )
    
    logger.info(f"PCA解释方差比例: {pca.explained_variance_ratio_}")
    logger.info(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # 可视化PCA结果
    
    # 1. 解释方差图
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 21), pca.explained_variance_ratio_)
    plt.xlabel('主成分')
    plt.ylabel('解释方差比例')
    plt.title('PCA解释方差比例')
    plt_path = os.path.join(args.save_dir, 'plots', 'pca_variance.png')
    plt.savefig(plt_path)
    plt.close()
    
    # 将图像添加到TensorBoard
    img = plt.imread(plt_path)
    writer.add_image('PCA/variance_ratio', np.transpose(img, (2, 0, 1)), 0)
    
    # 2. 累积解释方差图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('主成分数量')
    plt.ylabel('累积解释方差')
    plt.title('PCA累积解释方差')
    plt.grid(True)
    plt_path = os.path.join(args.save_dir, 'plots', 'pca_cumulative_variance.png')
    plt.savefig(plt_path)
    plt.close()
    
    # 将图像添加到TensorBoard
    img = plt.imread(plt_path)
    writer.add_image('PCA/cumulative_variance', np.transpose(img, (2, 0, 1)), 0)
    
    # 3. 特征向量在不同主成分上的投影
    plt.figure(figsize=(12, 8))
    
    # 按批次绘制点
    unique_batches = np.unique(batch_indices)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_batches)))
    
    for i, batch_id in enumerate(unique_batches):
        mask = np.array(batch_indices) == batch_id
        indices = np.where(mask)[0]
        
        # 展平特征向量，每个特征向量在特定的批次中
        for j in range(len(indices)):
            idx = indices[j]
            start_idx = idx * args.r
            end_idx = start_idx + args.r
            
            # 对每个特征向量的PCA结果进行绘制
            plt.scatter(
                pca_result[start_idx:end_idx, 0], 
                pca_result[start_idx:end_idx, 1],
                color=colors[i],
                alpha=0.7,
                label=f'Batch {batch_id}' if j == 0 else ""
            )
    
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.title('特征向量在前两个主成分上的投影')
    plt.legend()
    plt.grid(True)
    plt_path = os.path.join(args.save_dir, 'plots', 'pca_projection.png')
    plt.savefig(plt_path)
    plt.close()
    
    # 将图像添加到TensorBoard
    img = plt.imread(plt_path)
    writer.add_image('PCA/projection', np.transpose(img, (2, 0, 1)), 0)
    
    # 4. 批次之间的特征向量相似性分析
    # 计算不同批次的特征向量之间的相似性（余弦相似度）
    
    # 计算每个批次的平均特征向量
    batch_avg_vectors = {}
    for batch_id in unique_batches:
        mask = np.array(batch_indices) == batch_id
        indices = np.where(mask)[0]
        
        batch_vectors = []
        for idx in indices:
            start_idx = idx * args.r
            end_idx = start_idx + args.r
            batch_vectors.append(eigenvector_matrix[start_idx:end_idx])
        
        batch_avg_vectors[batch_id] = np.mean(np.vstack(batch_vectors), axis=0)
    
    # 计算余弦相似度矩阵
    similarity_matrix = np.zeros((len(unique_batches), len(unique_batches)))
    for i, batch_i in enumerate(unique_batches):
        for j, batch_j in enumerate(unique_batches):
            vec_i = batch_avg_vectors[batch_i]
            vec_j = batch_avg_vectors[batch_j]
            
            # 计算余弦相似度
            similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
            similarity_matrix[i, j] = similarity
    
    # 绘制相似度热图
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='余弦相似度')
    plt.title('不同批次特征向量间的余弦相似度')
    
    # 设置刻度标签
    batch_labels = [f'Batch {b}' for b in unique_batches]
    plt.xticks(range(len(unique_batches)), batch_labels, rotation=45)
    plt.yticks(range(len(unique_batches)), batch_labels)
    
    plt.tight_layout()
    plt_path = os.path.join(args.save_dir, 'plots', 'cosine_similarity.png')
    plt.savefig(plt_path)
    plt.close()
    
    # 将图像添加到TensorBoard
    img = plt.imread(plt_path)
    writer.add_image('PCA/cosine_similarity', np.transpose(img, (2, 0, 1)), 0)
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 记录总运行时间
    end_time = time.time()
    logger.info(f"总运行时间: {end_time - start_time:.2f} 秒")
    logger.info("实验完成！")

if __name__ == "__main__":
    main() 