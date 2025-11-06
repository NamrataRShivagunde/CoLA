
# tm10
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type cola \
    --model_config cola_configs/cola_60m.json \
    --lr 0.006 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 50 \
    --warmup_steps 10 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name cola-60m-wsd \
    --stable_steps 20


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

# rotating-cola-60m
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
    --run_name rotating-cola-60m 

CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type cola \
    --model_config cola_configs/cola_60m.json \
    --lr 0.003 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name rotating-cola-60m-lrpt003 \
    --save_every 10000

CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type cola \
    --model_config cola_configs/cola_60m.json \
    --lr 0.003 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 3000 \
    --weight_decay 0.02 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 1 \
    --run_name rotating-cola-60m-newhp \
    --save_every 10000


CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type cola \
    --model_config cola_configs/cola_60m.json \
    --lr 0.0038 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2500 \
    --weight_decay 0.015 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name rotating-cola-60m-align-basin \
    --save_every 10000


CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type rotating_cola \
    --model_config cola_configs/cola_60m.json \
    --lr 0.0045 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --rank 64 \
    --run_name rotating-cola-60m-highrank \
    --save_every 10000


CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc-per-node=1 main.py \
    --model_type cola \
    --model_config cola_configs/cola_60m.json \
    --lr 0.006 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 3000 \
    --weight_decay 0.02 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 1 \
    --run_name rotating-cola-60m-newhp-lr6 \
    --save_every 10000
