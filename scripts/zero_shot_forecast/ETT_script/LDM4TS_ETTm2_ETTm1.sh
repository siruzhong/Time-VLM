export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=3
model_name=LDM4TS
periodicity=24
batch_size=32
num_workers=32
learning_rate=0.001
seq_len=512
task_name=zero_shot_forecast

data=ETTm2
root_path=./dataset/ETT-small/
data_path=ETTm2.csv

target_data=ETTm1
target_root_path=./dataset/ETT-small/
target_data_path=ETTm1.csv

for pred_len in 96 192 336 720; do
    python -u run_ldm.py \
        --task_name $task_name \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id ETTm2_ETTm1_${seq_len}_${pred_len} \
        --model $model_name \
        --data $data \
        --features M \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --num_workers $num_workers \
        --target_data $target_data \
        --target_root_path $target_root_path \
        --target_data_path $target_data_path
done