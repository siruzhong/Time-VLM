export CUDA_VISIBLE_DEVICES=0

model_name=ScoreGrad
seq_len=512

for pred_len in 96 192 336 720
do

python -u run_ldm.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_${pred_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --batch_size 8 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1

done
