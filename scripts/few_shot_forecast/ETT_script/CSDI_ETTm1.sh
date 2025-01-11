export CUDA_VISIBLE_DEVICES=2
model_name=CSDI
seq_len=96

for pred_len in 96 192 336 720
do

python -u run_ldm.py \
  --task_name few_shot_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm1 \
  --percent 0.1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --periodicity 96 \
  --itr 3

done