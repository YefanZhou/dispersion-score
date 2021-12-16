# A Dataset-dispersion Perspective on Reconstruction versus Recognition in Single-view 3D Reconstruction Networks [3DV 2021]

**[Yefan Zhou](https://yefanzhou.github.io/), [Yiru Shen](https://scholar.google.com/citations?hl=en&user=-_Hy9z0AAAAJ), [Yujun Yan](https://scholar.google.com/citations?hl=en&user=5TQUP58AAAAJ), [Chen Feng](https://scholar.google.com/citations?hl=en&user=YeG8ZM0AAAAJ), [Yaoqing Yang](https://scholar.google.com/citations?user=LYvugWgAAAAJ&hl=en)**

[Paper (Arxiv)](https://arxiv.org/abs/2111.15158)

**SVR model can be disposed towards recognition (classification-based) or reconstruction by changing the dispersion extent of training data.**

![github_twitter_demo](https://github.com/YefanZhou/dispersion-score/blob/main/figs/github_twitter_demo.png)



Dispersion Score is a data-driven metric that is used to measure the internel machaism of single-view 3D reconstruction network: the tendency of network to perform recognition or reconstruction. It can also be used to diagnose training data and guide data augmentation as a heuristic.

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

Download provided synthetic dataset and customized ShapeNet renderings as following, or you may [build synthetic dataset](https://github.com/YefanZhou/dispersion-score/tree/main/dataset/synthetic_data) or [build renderings](https://github.com/YefanZhou/dispersion-score/tree/main/dataset/data) yourself.

```bash
bash download/download_data.sh
```

Manually download ShapeNet V1 (AtlasNet version): [pointclouds](https://drive.google.com/file/d/1MMCYOqSalz77dduKahqDEQKFP9aCvUCy/view?usp=sharing), [renderings](https://drive.google.com/file/d/153nd1oUd5ONnP8AoXaU8IZunskd5LEvB/view?usp=sharing) , and unzip the two files as following.

```bash
unzip ShapeNetV1PointCloud.zip -d ./dataset/data/
unzip ShapeNetV1Renderings.zip -d ./dataset/data/
```



## Experiments Results

Download our trained models:

```bash
bash download/download_checkpts.sh
```



### Experiments on Synthetic datasets:

#### Measure Dispersion Score (DS) and Visualize Measurements

```bash
python eval_scripts/eval_ds_synthetic.py --gpus [IDS OF GPUS TO USE]
```

Run the [notebook](https://github.com/YefanZhou/dispersion-score/blob/main/plot_scripts/synthetic_dataset.ipynb) to visualize the results and reproduce plots. 

#### Model Training

You could also train models from scratch as following instead of using trained models.

```bash
python train_scripts/train_synthetic.py --gpus [IDS OF GPUS TO USE]
```



------

### Experiments on ShapeNet:

#### Measure Dispersion Score (DS) and Visualize Measurements

```bash
# More dispersed training Images 
python eval_scripts/eval_ds_moreimgs.py --gpus [IDS OF GPUS TO USE]
```

```bash
# More dispersed training shapes 
python eval_scripts/eval_ds_moreshapes.py --gpus [IDS OF GPUS TO USE] 
```

Run the [notebook](https://github.com/YefanZhou/dispersion-score/blob/main/plot_scripts/shapenet_results.ipynb) to visualize the results and reproduce plots. 

#### Model Training

You could also train models from scratch as following instead of using trained models.

```bash
python train_scripts/train_more_imgs.py --gpus [IDS OF GPUS TO USE]
python train_scripts/train_more_shapes.py --gpus [IDS OF GPUS TO USE]
```



## Citation
We appreciate it if you would please cite the following paper if you found the repository useful for your work:
```
@article{zhou2021dataset,
  title={A Dataset-Dispersion Perspective on Reconstruction Versus Recognition in Single-View 3D Reconstruction Networks},
  author={Zhou, Yefan and Shen, Yiru and Yan, Yujun and Feng, Chen and Yang, Yaoqing},
  journal={arXiv preprint arXiv:2111.15158},
  year={2021}
}
```



