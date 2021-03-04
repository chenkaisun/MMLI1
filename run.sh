

for _plm_lr in 2e-5 5e-5; do
  for _model_type in tdg td t; do
    for _lr in 1e-4 1e-3; do
      for _max_grad_norm in 0 1; do
        for _g_dim in 128 256; do
          for _mult_mask in 0 1; do
            for _g_mult_mask in 0 1; do

              python main_re.py \
              --use_cache 1 \
              --batch_size 24 \
              --num_epoch 15 \
              --grad_accumulation_steps 1 \
              --plm_lr $_plm_lr \
              --lr $_lr \
              --model_type $_model_type \
              --g_dim $_g_dim \
              --patience 8 \
              --max_grad_norm $_max_grad_norm \
              --mult_mask $_mult_mask \
              --g_mult_mask $_g_mult_mask

            done
          done
        done
      done
    done
  done
done
