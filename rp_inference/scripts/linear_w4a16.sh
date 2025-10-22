export HF_HOME=~/workspace/hf_cache

model=$2
seed=0
tasks=mmlu
num_fewshot=none
eval_ppl=false

auto_dtype=true
custom_cuda=true
per_tensor=false # per-tensor quantization
quarot=false
rotate_mode=hadamard
rotate_kv=false
kv_quant_only=false
kv_tokenwise=false

# Linear precision
w_elem_format_linear=fp4_e2m1 # set none for bf16 baseline
a_elem_format_linear=none
block_size_linear=32
scale_bits_linear=8
w_scale_mode=0 # 0:PoT(Floor), 3: PoT(Round), 152: E5M2
a_scale_mode=0

# Attention Matmul
A_elem_format_matmul=none
B_elem_format_matmul=none
scale_bits_matmul=8
block_size_matmul=32
A_scale_mode=0
B_scale_mode=0

# LN (Not implemented)
w_elem_format_ln=none
a_elem_format_ln=none
scale_bits_ln=8
block_size_ln=32

# Head
w_elem_format_head=none
a_elem_format_head=none
scale_bits_head=8
block_size_head=32

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model=$model \
    --seed=$seed \
    --tasks=$tasks \
    --num_fewshot=$num_fewshot \
    --eval_ppl=$eval_ppl \
    --w_elem_format_linear=$w_elem_format_linear \
    --a_elem_format_linear=$a_elem_format_linear \
    --scale_bits_linear=$scale_bits_linear \
    --block_size_linear=$block_size_linear \
    --A_elem_format_matmul=$A_elem_format_matmul \
    --B_elem_format_matmul=$B_elem_format_matmul \
    --scale_bits_matmul=$scale_bits_matmul \
    --block_size_matmul=$block_size_matmul \
    --w_elem_format_ln=$w_elem_format_ln \
    --a_elem_format_ln=$a_elem_format_ln \
    --scale_bits_ln=$scale_bits_ln \
    --block_size_ln=$block_size_ln \
    --w_elem_format_head=$w_elem_format_head \
    --a_elem_format_head=$a_elem_format_head \
    --scale_bits_head=$scale_bits_head \
    --block_size_head=$block_size_head \
    --auto_dtype=$auto_dtype \
    --custom_cuda=$custom_cuda \
    --a_scale_mode=$a_scale_mode \
    --w_scale_mode=$w_scale_mode \
    --A_scale_mode=$A_scale_mode \
    --B_scale_mode=$B_scale_mode \
    --per_tensor=$per_tensor \
    --quarot=$quarot \
    --rotate_mode=$rotate_mode \
    --rotate_kv=$rotate_kv \
    --kv_quant_only=$kv_quant_only \
    --kv_tokenwise=$kv_tokenwise
