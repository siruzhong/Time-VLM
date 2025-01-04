export TOKENIZERS_PARALLELISM=false
model_name=TimeVLM
vlm_type=vilt
gpu=0
image_size=56
predictor_hidden_dims=128
periodicity=24
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=32
num_workers=32
learning_rate=0.001
seq_len=96
task_name=zero_shot_forecast

data=ETTm1
root_path=./dataset/ETT-small/
data_path=ETTm1.csv

target_data=ETTm2
target_root_path=./dataset/ETT-small/
target_data_path=ETTm2.csv

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --predictor_hidden_dims $predictor_hidden_dims \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_vlm $finetune_vlm \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers \
  --vlm_type $vlm_type \
  --target_data $target_data \
  --target_root_path $target_root_path \
  --target_data_path $target_data_path \

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --predictor_hidden_dims $predictor_hidden_dims \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_vlm $finetune_vlm \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers \
  --vlm_type $vlm_type \
  --target_data $target_data \
  --target_root_path $target_root_path \
  --target_data_path $target_data_path \

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --predictor_hidden_dims $predictor_hidden_dims \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_vlm $finetune_vlm \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers \
  --vlm_type $vlm_type \
  --target_data $target_data \
  --target_root_path $target_root_path \
  --target_data_path $target_data_path \

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --predictor_hidden_dims $predictor_hidden_dims \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_vlm $finetune_vlm \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers \
  --vlm_type $vlm_type \
  --target_data $target_data \
  --target_root_path $target_root_path \
  --target_data_path $target_data_path \