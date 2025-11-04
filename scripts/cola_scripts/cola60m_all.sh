
# tm10
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type cola \
    --model_config cola_configs/cola_60m.json \
    --lr 0.006 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name cola-60m-seed0 \
    --seed 0 


# tm11
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type cola \
    --model_config cola_configs/cola_60m.json \
    --lr 0.006 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name cola-60m-seed1024 \
    --seed 1024

# tm12
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type cola \
    --model_config cola_configs/cola_60m.json \
    --lr 0.006 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name cola-60m-seed100 \
    --seed 100

# tm13
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type cola \
    --model_config cola_configs/cola_60m.json \
    --lr 0.006 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name cola-60m-seed512 \
    --seed 512



# BASELINE 5 times with 5 seeds
