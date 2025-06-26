# Source code for Diffusion Experiments
Based on the code from [RPSW](https://github.com/khainb/RPSW).

## Installation
Python 3.9.12 is used for the experiments. The code is tested on Ubuntu 20.04.1 LTS.

Activate conda environment 
```bash
conda activate nonlinear-tsw
```

No need to setup data for CIFAR-10, as the code will download the dataset automatically.

## CIFAR-10 Training

For Tree-Sliced methods:
```bash
# TSW-SL:
torchrun --standalone --nproc_per_node=2 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_cltwd_4lines_linear_uniform --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4  --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_gen_mode gaussian_raw --twd_mass_division uniform --twd_delta 10 --twd_std 0.1 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "name" --save_ckpt_every 25 --twd_ftype linear --twd_radius 0.01

# Db-TSW:
torchrun --standalone --nproc_per_node=2 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_cltwd_4lines_linear --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4  --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_gen_mode gaussian_raw --twd_delta 10 --twd_std 0.1 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "name" --save_ckpt_every 25 --twd_ftype linear

# Db-TSW$^\perp$:
torchrun --standalone --nproc_per_node=2 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_cltwd_4lines_linear_orthogonal --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4  --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_gen_mode gaussian_orthogonal --twd_delta 10 --twd_std 0.1 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "name" --save_ckpt_every 25 --twd_ftype linear

# CircularTSW:
torchrun --standalone --nproc_per_node=2 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_cltwd_4lines_circular --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4  --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_gen_mode gaussian_raw --twd_delta 10 --twd_std 0.1 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "name" --save_ckpt_every 25 --twd_ftype circular --twd_radius 0.01

# SpatialTSW: h(y) = y + y ^ 3:

torchrun --standalone --nproc_per_node=2 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_cltwd_4lines_linearpow_1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4  --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_gen_mode gaussian_raw --twd_delta 10 --twd_std 0.1 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "name" --save_ckpt_every 25 --twd_ftype pow --twd_pow_beta 1

# CircularTSW with r = 0

torchrun --standalone --nproc_per_node=2 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_cltwd_4lines_circular_r0 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4  --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_gen_mode gaussian_raw --twd_delta 10 --twd_std 0.1 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "name" --save_ckpt_every 25 --twd_ftype circular_r0

# CircularTSW with r = 0, k = 1
torchrun --standalone --nproc_per_node=2 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_cltwd_1lines_circular_r0 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4  --lazy_reg 15 --loss cltwd --T 10000 --L 1 --twd_gen_mode gaussian_raw --twd_delta 10 --twd_std 0.1 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "name" --save_ckpt_every 25 --twd_ftype circular_r0
```

For Tree-Sliced with evaluation:
```bash
CUDA_VISIBLE_DEVICES=1 python3 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_cltwd --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
--num_process_per_node 1 --port 10026 \
--loss cltwd --T 1024 --L 10 --twd_delta 1 \
--ch_mult 1 2 2 2 --save_content \
--wandb_project_name "twd" --wandb_entity "your-username" --eval
```

For DDGAN
```bash
CUDA_VISIBLE_DEVICES=1 python3 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_ddgan --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --num_process_per_node 1 --loss gan \
--ch_mult 1 2 2 2 --save_content \
```

For SW
```bash
CUDA_VISIBLE_DEVICES=6 python3 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_sw --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --num_process_per_node 1 --port 10036 --loss sw --L 10000 \
--ch_mult 1 2 2 2 --save_content \
--wandb_project_name "twd" --wandb_entity "name"
```

For DSW
```bash
CUDA_VISIBLE_DEVICES=6 python3 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_dsw --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --num_process_per_node 1 --port 10036 --loss dsw --L 10000 \
--ch_mult 1 2 2 2 --save_content \
--wandb_project_name "twd" --wandb_entity "name"
```



For EBSW
```bash
python3 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --num_process_per_node 1 --loss ebsw --L 10000 \
--ch_mult 1 2 2 2 --save_content
```

For RPSW
```bash
python3 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --num_process_per_node 1 --loss rpsw --L 10000 \
--kappa 100 kappa2 1 --beta 10 --ch_mult 1 2 2 2 --save_content
```

For IWRPSW
```bash
python3 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --num_process_per_node 1 --loss iwrpsw --L 10000 \
--kappa 100 kappa2 1 --beta 10 --ch_mult 1 2 2 2 --save_content
```


#### CIFAR-10 Testing ####

DDGAN
```bash
python3 test_ddgan.py --dataset cifar10 --exp ddgan_cifar10_cltwd_4lines_linear_uniform --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --compute_fid \
--wandb_project_name "twd" --wandb_entity "entity" --max_epoch_id 1800 --min_epoch_id 0
```
