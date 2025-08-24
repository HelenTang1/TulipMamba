
#!/bin/bash

args=(
    --eval
    --mc_drop
    --noise_threshold 0.03
    --model_select tulip_base
    --pixel_shuffle
    --circular_padding
    --patch_unmerging
    --log_transform
    # Dataset
    --dataset_select kitti
    --data_path_low_res ./dataset/kitti_64_2048/
    --data_path_high_res ./dataset/kitti_64_2048/
    # --data_path_low_res ./dataset/kitti_exp/
    # --data_path_high_res ./dataset/kitti_exp/
    # --save_pcd
    # WandB Parameters
    --run_name tulip_base_uni_mamba
    # --entity myentity
    # --wandb_disabled
    --project_name kitti_evaluation
    --output_dir ./experiment/kitti/tulip_base/checkpoint-900.pth
    --img_size_low_res 16 2048
    --img_size_high_res 64 2048
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    --save_pcd
    --device cuda:9
    )

python main_lidar_upsampling.py "${args[@]}"
