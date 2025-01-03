export TOKENIZERS_PARALLELISM=false
gpu=0

task_name=zero_shot_forecast
model=TimeVLM
data=ETTm2
target_data=ETTh2
target_data_path=ETTh2.csv
features=M
seq_len=96
label_len=48
pred_lens=(96 192 336 720)
d_model=128
n_heads=8
e_layers=2
d_layers=1
d_ff=768
expand=2
d_conv=4
factor=3
embed=timeF
distil=True
des=Exp
ii=0

predictor_hidden_dims=128

for pred_len in "${pred_lens[@]}"; do
  model_id="ETTh1_96_${pred_len}"
  
  setting=$(python -c "print('long_term_forecast_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_0'.format(
      '${model_id}',
      '${model}',
      '${data}',
      '${features}',
      '${seq_len}',
      '${label_len}',
      '${pred_len}',
      '${d_model}',
      '${n_heads}',
      '${e_layers}',
      '${d_layers}',
      '${d_ff}',
      '${expand}',
      '${d_conv}',
      '${factor}',
      '${embed}',
      '${distil}',
      '${des}',
      ${ii}
  ))")

  pretrained_model_path="./checkpoints/${setting}/checkpoint.pth"

  python -u run.py \
    --task_name $task_name \
    --is_training 0 \
    --model_id $model_id \
    --model $model \
    --data $target_data \
    --root_path ./dataset/ETT-small/ \
    --data_path $target_data_path \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --d_model $d_model \
    --n_heads $n_heads \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --d_ff $d_ff \
    --expand $expand \
    --d_conv $d_conv \
    --factor $factor \
    --embed $embed \
    --des $des \
    --pretrained_model_path $pretrained_model_path \
    --predictor_hidden_dims $predictor_hidden_dims \
    --gpu $gpu
done