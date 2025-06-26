export CUDA_VISIBLE_DEVICES=4
python GradientFlow_test.py --num_iter 5000 --L 100 --n_lines 2 --lr_sw 5e-3 --lr_tsw_sl 0.05 --delta 1.5 --p 2 --dataset_name "gaussian_20d_small_v" --std 0.001 --num_seeds 1
