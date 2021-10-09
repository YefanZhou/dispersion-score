import glob
from os.path import join
import random

actual_path = "data/filelists"
expect_path = "data/origin_filelists"
r2n2_path = '../../3D-R2N2/data/filelists'
onet_path = '../../occupancy_networks/data/ShapeNet'

train_lst_path = join(actual_path, 'train.lst')
test_lst_path = join(actual_path, 'test.lst')
id_list = []
with open(train_lst_path, 'r') as f:
    train_lst = f.read().splitlines()

with open(test_lst_path, 'r') as f:
    test_lst = f.read().splitlines()


train_paths = glob.glob(join(r2n2_path, '*_train.lst'))

for train_path in train_paths:
    id_list.append(train_path.split('/')[-1].split('_')[0])

total_amount = 0
total_train = 0
total_test = 0
for id_f in id_list:
    id_train_test = glob.glob(join(actual_path, f'{id_f}_*.lst'))
    id_train_test.sort()
    with open(id_train_test[0], 'r') as f:
        id_test_lst = f.read().splitlines()
    with open(id_train_test[1], 'r') as f:
        id_train_lst = f.read().splitlines()

    id_lst = id_test_lst + id_train_lst
    total_amount += len(id_lst)
    total_train += len(id_train_lst)
    total_test += len(id_test_lst)

    fo = open(join(actual_path, "id_lst", f"{id_f}.lst"), "w")
    for f_id in id_lst:
        fo.write(f_id + '\n')

print(total_amount, total_train, total_test)
    




        
