# TulipMamba
A Small Project for NYCU DLLab course, 2025 Summar
Reference: [TULIP: Transformer for Upsampling of LiDAR Point Clouds: A framework for LiDAR upsampling using Swin Transformer \(CVPR2024\)](https://arxiv.org/abs/2312.06733)

Git of TULIP: https://github.com/ethz-asl/TULIP.git

# Environment
- python=3.8
- torch=1.13.0+cu117
- torchaudio=0.13.0+cu117
- torchvision=0.14.0+cu117
- timm=0.6.13
- install requirements for TULIP
- build from source: causal-conv1d
- build from source: mamba

## Dataset:
Use only "City" data in [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php?type=city). Download script:
```bash=
./dataset/kitti/raw_data_downloader.sh
```
Preprocess Point Cloud to Range image with 2 different sizes: (64, 1024) and (64, 2048). 

Run preprocess script:
```bash=
./run_sample_kitti_dataset.sh
```

Data Statistic:
| Traing Data | Validation Data |
|-------------|-----------------|
|    2811     |   528           |

## Trained Models

- [x] tulip_base
- [x] tulip_base_mamba
- [x] tulip_base_mamba_crsatten
- [x] tulip_base_mamba_crsmamba

```bash=
./run_train_<model_name>.sh
```

### Hyperparameters:
- batch_size=8
- epochs=900
- lr=5e-4
- weight_decay=0.01
- warmup_epochs=60
### Model parameters
- model_select
    - tulip_base
    - tulip_base_mamba
    - tulip_base_mamba_crsatten
    - tulip_base_mamba_crsmamba
- pixel_shuffle 
- circular_padding 
- log_transform 
- patch_unmerging 
### Input settings
- dataset_select=kitti
- window_size=2 8
- patch_size=1 4
- in_chans=1
### input (64, 2048) for tulip_base, tulip_base_mamba, tulip_base_mamba_crsmamba
- data_path_low_res ./dataset/kitti_64_2048/
- data_path_high_res ./dataset/kitti_64_2048/
- img_size_low_res 16 2048
- img_size_high_res 64 2048

### input (64, 1024) for tulip_base_mamba_crsatten (because of hardware constraint )
- data_path_low_res ./dataset/kitti_64_1024/
- data_path_high_res ./dataset/kitti_64_1024/
- img_size_low_res 16 1024
- img_size_high_res 64 1024

## Evaluate Model
```bash=
./run_eval_<model_name>.sh
```

## Get the Evaluation Summary
Run: Summary.ipynb

## Get Model size Statistic
```bash=
./run_params_flops.sh
```
Ouput:
```
=================================
Input size:  (16, 1024)
Model: tulip_base_mamba_crsatten
Params: total 32.56M, trainable 32.56M
MACs (1 input): 19.76 GMac  | FLOPs ~ 39.51 GFLOPs
=================================
Input size:  (16, 2048)
Model: tulip_base_mamba_crsmamba
Params: total 33.07M, trainable 33.07M
MACs (1 input): 26.53 GMac  | FLOPs ~ 53.06 GFLOPs
=================================
Input size:  (16, 2048)
Model: tulip_base_mamba
Params: total 31.97M, trainable 31.97M
MACs (1 input): 23.99 GMac  | FLOPs ~ 47.98 GFLOPs
=================================
Input size:  (16, 2048)
Model: tulip_base
Params: total 27.15M, trainable 27.15M
MACs (1 input): 15.41 GMac  | FLOPs ~ 30.82 GFLOPs
=================================
Input size:  (16, 1024)
Model: tulip_base_mamba
Params: total 31.97M, trainable 31.97M
MACs (1 input): 12.00 GMac  | FLOPs ~ 23.99 GFLOPs
```