export CUDA_VISIBLE_DEVICES=3
model_name=ScoreGrad
seq_len=512

for pred_len in 96 192 336 720
do

python -u run_ldm.py \
  --task_name few_shot_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --percent 0.1 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --e_layers 1 \
  --des 'Exp' \
  --periodicity 24 \
  --learning_rate 0.001 \
  --itr 3

done