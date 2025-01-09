gpu=3
model_name=TimeLLM
learning_rate=0.01
batch_size=24
d_ff=128
seq_len=512

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_512_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 256 \
  --d_ff 256 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --batch_size $batch_size \
  --d_ff $d_ff \
  --gpu $gpu

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_512_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 256 \
  --d_ff 256 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --batch_size $batch_size \
  --d_ff $d_ff \
  --gpu $gpu

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_512_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 256 \
  --d_ff 256 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --batch_size $batch_size \
  --d_ff $d_ff \
  --gpu $gpu

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_512_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 256 \
  --d_ff 256 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --batch_size $batch_size \
  --d_ff $d_ff \
  --gpu $gpu