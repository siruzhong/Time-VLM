export CUDA_VISIBLE_DEVICES=2

model_name=LightTS
seq_len=96

for pred_len in 96 192 336 720
do

python -u run_ldm.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_${pred_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --percent 0.1 \
  --des 'Exp' \
  --itr 1

done
