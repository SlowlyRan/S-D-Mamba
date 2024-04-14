export CUDA_VISIBLE_DEVICES=2

model_name=S_Mamba

python -u test_run.py \
  --is_training 1 \
  --root_path ./kaggle/input/solar-al/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1
