#!/bin/bash

# 设置默认值
MINI_HESSIAN_BATCH_SIZE=200
HESSIAN_BATCH_SIZE=200
SEED=42
BATCH_NORM=true
RESIDUAL=true
CUDA=true
RESUME="checkpoints/resnet20_cifar10.pth"

# 显示使用说明
function show_usage {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  --mini-hessian-batch-size  mini hessian批量大小 (默认: 200)"
    echo "  --hessian-batch-size       hessian批量大小 (默认: 200)"
    echo "  --seed                     用于复现结果的随机种子 (默认: 42)"
    echo "  --no-batch-norm            在ResNet中不使用batch norm (默认启用batch norm)"
    echo "  --no-residual              不使用残差连接 (默认启用残差)"
    echo "  --no-cuda                  不使用GPU (默认使用GPU)"
    echo "  --resume PATH              检查点文件路径 (必填)"
    echo "  --help                     显示此帮助信息"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mini-hessian-batch-size)
            MINI_HESSIAN_BATCH_SIZE="$2"
            shift 2
            ;;
        --hessian-batch-size)
            HESSIAN_BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --no-batch-norm)
            BATCH_NORM=false
            shift
            ;;
        --no-residual)
            RESIDUAL=false
            shift
            ;;
        --no-cuda)
            CUDA=false
            shift
            ;;
        --resume)
            RESUME="$2"
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

# 检查必填参数
if [ -z "$RESUME" ]; then
    echo "错误: 必须提供--resume参数指定检查点文件路径"
    show_usage
    exit 1
fi

# 构建命令
CMD="export CUDA_VISIBLE_DEVICES=0; python example_pyhessian_analysis.py --mini-hessian-batch-size $MINI_HESSIAN_BATCH_SIZE --hessian-batch-size $HESSIAN_BATCH_SIZE --seed $SEED --resume $RESUME"

# 添加可选标志参数
if [ "$BATCH_NORM" = false ]; then
    CMD="$CMD --batch-norm"
fi

if [ "$RESIDUAL" = false ]; then
    CMD="$CMD --residual"
fi

if [ "$CUDA" = false ]; then
    CMD="$CMD --cuda"
fi

# 显示并执行命令
echo "运行命令: $CMD"
eval $CMD