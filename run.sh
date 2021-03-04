python main.py --debug

#for _dataset in webkb_wisconsin webkb_cornell webkb_washington; do
#  for _mp in 0.1 0.2 0.5 0.3; do
#    for _model in FREQ MF ZERO; do
#      echo _dataset
#      python main.py \
#            --seed 42 \
#            --num_folds 5 \
#            --lr 0.002 \
#            --joint 0 \
#            --dataset $_dataset \
#            --missing_proportion $_mp \
#            --patience 50 \
#            --epochs 1000 \
#            --model $_model
#    done
#  done
#done
#