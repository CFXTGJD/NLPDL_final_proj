wandb offline

#When training from the original mT5 model:
python finetune_mt5_train_and_test.py --dataset "jec_qa_CA" \
    --run_name "jec_qa_CA_after_jec_qa_CA" --project_name "jec_qa_CA"

#When training from some finetuned task: 
python finetune_mt5_train_and_test.py --dataset "jec_qa_CA" \
    --run_name "jec_qa_CA_after_jec_qa_CA" --project_name "jec_qa_CA" \
    --load_from_checkpoint True \
    --load_path "/ceph/home/liuxingyu/NLP/final/mT5/lightning_logs/1e6xekjo/checkpoints/checkpoints/mt5-small-dataset-epoch=01-val_loss=0.47-jec_qa_CA-jec_qa_CA_first-v1.ckpt"
