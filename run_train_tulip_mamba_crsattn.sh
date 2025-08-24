
#!/bin/bash


args=(
    --batch_size 8
    --epochs 1700
    --num_workers 2
    --lr 5e-4
    --weight_decay 0.01
    --warmup_epochs 60
    # Model parameters
    --model_select tulip_base_mamba_crsatten
    --pixel_shuffle # improve
    --circular_padding # improve
    --log_transform # improve
    --patch_unmerging # improve
    # Dataset
    --dataset_select kitti
    --data_path_low_res ./dataset/kitti_64_1024/
    --data_path_high_res ./dataset/kitti_64_1024/
    # WandB Parameters
    --run_name tulip_base_mamba_crsatten
    # --entity myentity
    # --wandb_disabled
    --project_name experiment_kitti
    --output_dir ./experiment/kitti/tulip_base_mamba_crsatten
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

# real batch size in training = batch_size * nproc_per_node
torchrun --nproc_per_node=4 main_lidar_upsampling.py "${args[@]}"
