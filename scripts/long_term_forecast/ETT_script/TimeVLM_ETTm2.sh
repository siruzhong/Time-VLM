model_name=TimeVLM
gpu=1
predictor_hidden_dims=128

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_48 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1 \
  --gpu $gpu \
  --predictor_hidden_dims $predictor_hidden_dims


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_192_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1 \
  --gpu $gpu \
  --predictor_hidden_dims $predictor_hidden_dims

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_288_144 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1 \
  --gpu $gpu \
  --predictor_hidden_dims $predictor_hidden_dims

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_384_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1 \
  --gpu $gpu \
  --predictor_hidden_dims $predictor_hidden_dims