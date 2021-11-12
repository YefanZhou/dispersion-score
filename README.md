# dispersion-score



Official implementation of our 3DV Paper *A Dataset-dispersion Perspective on Reconstruction versus Recognition in Single-view 3D Reconstruction Networks* 

## Installation

To install dispersion-score and develop locally:

- PyTorch version >= 1.6.0
- Python version = 3.6

```bash
conda create -n dispersion_score python=3.6
conda activate dispersion_score
git clone https://github.com/YefanZhou/dispersion-score.git
cd dispersion-score
chmod +x setup.sh 
./setup.sh
```

------

## Dataset

Synthetic dataset: you may download the provided data as following or [build it yourself](https://github.com/YefanZhou/dispersion-score/tree/main/dataset/synthetic_data).

```bash
bash dataset/download_synthetic_data.sh
```

ShapeNet V1 (AtlasNet version): [pointclouds](https://drive.google.com/file/d/1MMCYOqSalz77dduKahqDEQKFP9aCvUCy/view?usp=sharing), [renderings](https://drive.google.com/file/d/153nd1oUd5ONnP8AoXaU8IZunskd5LEvB/view?usp=sharing)  

ShapeNet V1 [customized renderings](https://drive.google.com/file/d/1Y2nAktYbeRJ23bhjqZp3c4kh5P8oBYdp/view?usp=sharing). You may download the provided data as following or [build it yourself](https://github.com/YefanZhou/dispersion-score/tree/main/dataset/data).

Manually download the three files to main folder.

```bash
unzip ShapeNetV1PointCloud.zip -d ./dataset/data/
unzip ShapeNetV1Renderings.zip -d ./dataset/data/
unzip ShapeNetV1RenderingAnglelimit.zip -d ./dataset/data/
```

## Experiments on Synthetic datasets:

### Model Training 

Train models on 16 train sets of synthetic datasets.

```bash
python train_scripts/train_synthetic.py --gpus [IDS OF GPUS TO USE]
```

You may also download our [trained models](https://drive.google.com/file/d/1h3mZUMueGfrhPTZo1BoNNYLyPymlpZb7/view?usp=sharing).

```bash
mkdir -p log/train_synthetic
unzip toydata_checkpoints.zip -d ./log/train_synthetic/
```

### Measure Dispersion Score (DS)

```bash
python eval_scripts/eval_ds_synthetic.py --gpus [IDS OF GPUS TO USE]
```

### Visualize the Results

Run the [notebook](https://github.com/YefanZhou/dispersion-score/blob/main/plot_scripts/synthetic_dataset.ipynb).

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





