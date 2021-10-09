import os
import glob
from os.path import join

path = "filelists"

filelsts = os.listdir(path)
print(filelsts)

for catf in filelsts:
    catid = catf.split('.')[0]
    os.system(f"unzip ../ShapeNetV1/ShapeNetCore.v1/{catid}.zip -d ../ShapeNetV1/ShapeNetCore.v1")