# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer galore_adamw 


# baseline 60M - cosine
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.001 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --scheduler cosine



# baseline 60M - wsd tm1
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.001 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --stable_steps 7000 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --scheduler warm_stable_decay