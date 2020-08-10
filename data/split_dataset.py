"""
 Created by Myung-Joon Kwon
 corundum240@gmail.com
 Aug 4, 2020
"""
import project_config
import os
import random
from pathlib import Path
root = project_config.dataset_paths['djpeg']
imlist = []  # format: im.jpg,0 (0 for single, 1 for double)
train = []
val = []
val_ratio = 0.02
for file in os.listdir(root / 'single'):
    if not file.lower().endswith(".jpg"):
        raise TypeError
    imlist.append('single/' + file + ',0')
print(len(imlist))
can_val = random.sample(imlist, k=int(len(imlist)*val_ratio))
val.extend(can_val)
for i in can_val:
    imlist.remove(i)
train.extend(imlist)
print(len(train), len(val))
imlist=[]
for file in os.listdir(root / 'double'):
    if not file.lower().endswith(".jpg"):
        raise TypeError
    imlist.append('double/' + file + ',1')
print(len(imlist))
can_val = random.sample(imlist, k=int(len(imlist)*val_ratio))
val.extend(can_val)
for i in can_val:
    imlist.remove(i)
train.extend(imlist)
print(len(train), len(val))
with open(project_config.project_root / "train.txt", "w") as f:
    f.write('\n'.join(train) + '\n')
with open(project_config.project_root / "val.txt", "w") as f:
    f.write('\n'.join(val) + '\n')
