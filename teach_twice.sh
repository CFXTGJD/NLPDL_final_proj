#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=sbatch_example
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time 12:00:00

wandb offline
python finetune_mt5_train_and_test.py --dataset "jec_qa_CA" \
    --run_name "jec_qa_CA_after_jec_qa_CA" --project_name "jec_qa_CA" \
    --load_from_checkpoint True \
    --load_path "/ceph/home/liuxingyu/NLP/final/mT5/lightning_logs/1e6xekjo/checkpoints/checkpoints/mt5-small-dataset-epoch=01-val_loss=0.47-jec_qa_CA-jec_qa_CA_first-v1.ckpt"