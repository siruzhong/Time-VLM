export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
model_name=LDM4TS
seq_len=96

for pred_len in 192
do

python -u run_ldm.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --percent 0.1 \
  --batch_size 8 \
  --num_workers 4 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --periodicity 24 \
  --itr 1

done