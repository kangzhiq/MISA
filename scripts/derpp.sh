#!/bin/bash

#SBATCH -J DP_Siblurry_CIFAR_500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH -t 6-0
#SBATCH -o %x_%j_%a.log
#SBATCH -e %x_%j_%a.err

date
ulimit -n 65536
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

conda --version
python --version

# CIL CONFIG
NOTE="Derpp-er500" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="derpp" # 
DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet, imagenet-r
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS=1
echo "SEEEDS="$SEEDS

OPT="adam"

if [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="vit" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=256

elif [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="vit" EVAL_PERIOD=1000
    BATCHSIZE=256; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=256

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=500 ONLINE_ITER=3
    MODEL_NAME="vit" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100

elif [ "$DATASET" == "imagenet-r" ]; then
    MEM_SIZE=500 ONLINE_ITER=3
    MODEL_NAME="vit" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100


else
    echo "Undefined setting"
    exit 1
fi

echo "Batch size $BATCHSIZE  onlin iter $ONLINE_ITER"
for RND_SEED in $SEEDS
do
    python -W ignore main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir /local_datasets \
    --note $NOTE --eval_period $EVAL_PERIOD --transforms autoaug --memory_epoch $MEMORY_EPOCH --n_worker 8 --rnd_NM \

done

