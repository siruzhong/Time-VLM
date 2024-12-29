export TOKENIZERS_PARALLELISM=false
model_name=TimeVLM
gpu=2
image_size=224
predictor_hidden_dims=128
periodicity=24
norm_const=0.4
three_channel_image=True
finetune_clip=False
batch_size=8
num_workers=32
learning_rate=0.001

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --predictor_hidden_dims $predictor_hidden_dims \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_clip $finetune_clip \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --predictor_hidden_dims $predictor_hidden_dims \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_clip $finetune_clip \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --predictor_hidden_dims $predictor_hidden_dims \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_clip $finetune_clip \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --predictor_hidden_dims $predictor_hidden_dims \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_clip $finetune_clip \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers