# dispersion-score



Official implementation of our 3DV Paper *A Dataset-dispersion Perspective on Reconstruction versus Recognition in Single-view 3D Reconstruction Networks* 

## Installation

To install dispersion-score and develop locally:

- PyTorch version >= 1.6.0
- Python version = 3.6

```bash
git clone https://github.com/YefanZhou/dispersion-score.git
cd dispersion-score
chmod +x setup.sh 
```

------

## Dataset

Synthetic dataset: you may download our provided data or build it yourself.

```bash
bash dataset/download_synthetic_data.sh
```

ShapeNet V1: [pointclouds](https://drive.google.com/file/d/1MMCYOqSalz77dduKahqDEQKFP9aCvUCy/view?usp=sharing), [renderings](https://drive.google.com/file/d/153nd1oUd5ONnP8AoXaU8IZunskd5LEvB/view?usp=sharing)  

ShapeNet V1: customized renderings, you may download our provided data or build it yourself.

Manually download the three files to main folder.

```bash
unzip ShapeNetV1PointCloud.zip -d ./dataset/data/
unzip ShapeNetV1Renderings.zip -d ./dataset/data/
unzip ShapeNetV1RenderingAnglelimit.zip -d ./dataset/data/
```

## Experiments on Synthetic datasets:

### Model Training 

To train models on 16 train sets of  synthetic datasets.

```bash
python -m train_cmd.train_synthetic
```

The checkpoints is saved to **"log/toydata/*/checkpoint.pt"** ,  predicted shapes of test set **"log/toydata/*/prediction.npy"**. 

Each training takes 20 minitues and use 4.3GB GPU memory. 

### Measure Dispersion Score (DS)

```bash
python -m eval_cmd.eval_ds_synthetic
```

The input DS are saved to **"eval/eval_results/eval_ds_synthetic/checkpoints_1**", 

The output DS are saved to **"eval/eval_results/eval_ds_synthetic/checkpoints_pred**"

------



## Experiments on ShapeNet:

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





