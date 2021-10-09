# dispersion-score

## Introduction

Official implementation of our 3DV Paper *A Dataset-dispersion Perspective on Reconstruction versus Recognition in Single-view 3D Reconstruction Networks* 

## Setup

```bash
virtualenv -p python3.6 svrds_venv
source svrds_venv/bin/activate
chmod +x setup.sh download_data.sh
# Setup environment 
./setup.sh
# Download ShapeNet dataset.
./download_data.sh
```

------

## Experiments on Synthetic datasets:

### Building Synthetic dataset (Interpolation between cube and sphere) 

1. **Mesh generation**

   Ubuntu 18.04, Install blender 2.79.

   Open **dataset/synthetic_data/cube_shpere_interpolation.blend** from screen. Run the script.

2. **Pointcloud sampling**

```bash
cd dataset/synthetic_data
python pointcloud_sample.py
```

### Model Training 

To train models on 16 train sets of  synthetic datasets. **( Section 4.2 )**

```bash
python -m train_cmd.train_synthetic
```

To monitor the training procedure

```bash
tensorboard --logdir "log/tensorboard/train_synthetic"
```

The checkpoints is saved to **"log/toydata/*/checkpoint.pt"** , predicted shapes of test set **"log/toydata/*/prediction.npy"**. 

Each training takes 20 minitues and use 4.3GB GPU memory. 

### Measure Dispersion Score (DS)

```bash
python -m eval_cmd.eval_ds_synthetic
```

The input DS are saved to **"eval/eval_results/eval_ds_synthetic/checkpoints_1**", output DS are saved to **"eval/eval_results/eval_ds_synthetic/checkpoints_pred**"



------



## Experiments on ShapeNet:

### Dataset:

#### 	Download 3D R2N2 dataset:

Download the dataset:

```bash
chmod +x download_data.sh
./download_data.sh
```

#### 	Building extra image datasets with varying angle range of viewpoints

To generate the dataset:

- First download **ShapeNet v1** following the link: https://github.com/laughtervv/DISN#download-shapenetcorev1

- Then Install blender 2.79 and use bundled-python pip install numpy and opencv. Follow the instructions here: 

  https://blender.stackexchange.com/questions/56011/how-to-install-pip-for-blenders-bundled-python

```bash
wget https://www.blender.org/download/Blender2.79/blender-2.79-linux-glibc219-x86_64.tar.bz2
tar -xvf blender-2.79-linux-glibc219-x86_64.tar.bz2 /usr/local/

## Install pip for blener's bundled python, and then use it install numpy and opencv
/usr/local/blender-2.79-linux-glibc219-x86_64/2.79/python/local/lib/python3.5/dist-packages/pip install numpy opencv-python
```

- Then:

```bash
cd dataset/ShapenetRender_more_variation
python blender_batch_angle_cmd.py --model_root_dir {ShapeNetV1 root dir} --blender_path /usr/local/blender-2.79b-linux-glibc219-x86_64/blender --render_root_dir ../data/ShapeNetV1RenderingAnglelimit
```

It takes about 21 hours to generate 7 image datasets. 

### Model Training and Measure Dispersion Score (DS)

- More dispersed training Images 

```bash
## Training: Checkpoints saved to 'log/more_imgs'
python -m train_cmd.train_more_imgs
## Evaluation of DS, chamfer
python -m eval_cmd.eval_ds_moreimgs
```

- More dispersed training shapes

```bash
## Checkpoints saved to 'log/more_shapes'
python -m train_cmd.train_more_shapes
## Evaluation of DS, chamfer
python -m eval_cmd.eval_ds_moreshapes
```

- More training samples

```bash
## Checkpoints saved to 'log/nviews'
python -m train_cmd.train_nviews
## Evaluation of DS, chamfer
python -m eval_cmd.eval_ds_nviews
```

```bash
## Monitor training 
tensorboard --logdir log/tensorboard/more_imgs
tensorboard --logdir log/tensorboard/more_shapes
tensorboard --logdir log/tensorboard/nviews
```



**Complete codebase will release soon.**