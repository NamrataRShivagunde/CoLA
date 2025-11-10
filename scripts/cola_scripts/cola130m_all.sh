
# 130M cola with cosine scheduler
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type cola \
    --model_config cola_configs/cola_130m.json \
    --lr 0.003 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name cola-130m-cosine \
    --save_every 2000 \
    --scheduler cosine


# 130M cola with WSD

CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type cola \
    --model_config cola_configs/cola_130m.json \
    --lr 0.003 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name cola-130m-wsd \
    --save_every 2000 \
    --scheduler warm_stable_decay \
    --stable_steps 14000