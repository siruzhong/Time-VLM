export CUDA_VISIBLE_DEVICES=2
model_name=DiffusionTS

for pred_len in 96 192 336 720
do

python -u run_ldm.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$pred_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --batch_size 16 \
  --num_workers 8 \
  --features M \
  --seq_len $pred_len \
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