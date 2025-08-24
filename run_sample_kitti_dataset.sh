
args=(
   --image_rows 64
   --image_cols 1024
   --num_data_train 20000
   --num_data_val 2500
   --output_path_name_train train
   --output_path_name_val val
   --input_path ./dataset/kitti
   --output_path ./dataset/kitti_64_1024
   --create_val
   )

python ./sample_kitti_dataset.py "${args[@]}"

args=(
    --image_rows 64
    --image_cols 2048
    --num_data_train 20000
    --num_data_val 2500
    --output_path_name_train train
    --output_path_name_val val
    --input_path ./dataset/kitti
    --output_path ./dataset/kitti_64_2048
    --create_val
    )

python ./sample_kitti_dataset.py "${args[@]}"
