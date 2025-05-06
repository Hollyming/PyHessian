#!/bin/bash

# 设置默认值
BATCH_SIZE=128
HESSIAN_BATCH_SIZE=128
MINI_HESSIAN_BATCH_SIZE=128
EPOCHS=2
LR=0.001
R=10
SAVE_INTERVAL=1
RESUME="./checkpoints/resnet20_cifar10.pkl"
CIFAR10C_PATH="../data/CIFAR-10-C"
NOISE_TYPE="gaussian_noise"
CORRUPTION_LEVEL=5
SEED=42
CUDA=true
SAVE_DIR="./hessian_analysis"
GPU_ID=5  # 默认使用的GPU ID

# 显示使用说明
function show_usage {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  --batch-size              微调的批量大小 (默认: 128)"
    echo "  --hessian-batch-size      计算Hessian的批量大小 (默认: 128)"
    echo "  --mini-hessian-batch-size 计算Hessian的mini批量大小 (默认: 128)"
    echo "  --epochs                  微调的epoch数 (默认: 2)"
    echo "  --lr                      学习率 (默认: 0.001)"
    echo "  --r                       计算前r个特征值/特征向量 (默认: 10)"
    echo "  --save-interval           保存Hessian的batch间隔 (默认: 10)"
    echo "  --resume                  预训练模型路径 (默认: ./checkpoints/resnet20_cifar10.pkl)"
    echo "  --cifar10c-path           CIFAR10-C数据集路径 (默认: ./data/CIFAR-10-C)"
    echo "  --noise-type              CIFAR10-C噪声类型 (默认: gaussian_noise)"
    echo "                            可选: gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur,"
    echo "                                  motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,"
    echo "                                  elastic_transform, pixelate, jpeg_compression"
    echo "  --corruption-level        CIFAR10-C噪声级别(1-5) (默认: 5)"
    echo "  --seed                    随机种子 (默认: 1)"
    echo "  --no-cuda                 不使用GPU (默认使用GPU)"
    echo "  --save-dir                结果保存目录 (默认: ./hessian_analysis)"
    echo "  --help                    显示此帮助信息"
    echo ""
    echo "注意: CIFAR10-C数据集将在首次运行时自动下载 (约1.1GB)，请确保有足够的磁盘空间和网络连接"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --hessian-batch-size)
            HESSIAN_BATCH_SIZE="$2"
            shift 2
            ;;
        --mini-hessian-batch-size)
            MINI_HESSIAN_BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --r)
            R="$2"
            shift 2
            ;;
        --save-interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --cifar10c-path)
            CIFAR10C_PATH="$2"
            shift 2
            ;;
        --noise-type)
            NOISE_TYPE="$2"
            shift 2
            ;;
        --corruption-level)
            CORRUPTION_LEVEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --no-cuda)
            CUDA=false
            shift
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "错误: 未知选项 $1"
            show_usage
            exit 1
            ;;
    esac
done

# 检查pip依赖
echo "检查依赖..."
pip install --quiet tqdm requests scikit-learn matplotlib

# 构建命令
CMD="export CUDA_VISIBLE_DEVICES=$GPU_ID; python finetune_and_analyze_hessian.py --batch-size $BATCH_SIZE --hessian-batch-size $HESSIAN_BATCH_SIZE --mini-hessian-batch-size $MINI_HESSIAN_BATCH_SIZE --epochs $EPOCHS --lr $LR --r $R --save-interval $SAVE_INTERVAL --resume $RESUME --cifar10c-path $CIFAR10C_PATH --noise-type $NOISE_TYPE --corruption-level $CORRUPTION_LEVEL --seed $SEED --save-dir $SAVE_DIR"

# 添加可选标志参数
if [ "$CUDA" = false ]; then
    CMD="$CMD --cuda"
fi

# 显示并执行命令
echo "注意: 如果CIFAR10-C数据集不存在，将自动下载 (约1.1GB)，可能需要一些时间"
echo "运行命令: $CMD"
eval $CMD 