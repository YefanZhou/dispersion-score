## 	Building ShapeNet rendering datasets with varying angle range of viewpoints

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