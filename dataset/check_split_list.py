import glob
from os.path import join
import random

actual_path = "data/filelists"
expect_path = "data/origin_filelists"
r2n2_path = '../../3D-R2N2/data/filelists'
onet_path = '../../occupancy_networks/data/ShapeNet'


# actual_test_paths = glob.glob(join(actual_path, '*_test.lst'))
# actual_test_paths.sort()
# expect_test_paths = glob.glob(join(expect_path, '*_test.lst'))
# expect_test_paths.sort()
# r2n2_test_paths = glob.glob(join(r2n2_path, '*_test.lst'))
# r2n2_test_paths.sort()
# r2n2_train_paths = glob.glob(join(r2n2_path, '*_train.lst'))
# r2n2_train_paths.sort()

# onet_test_paths = glob.glob(join(onet_path, '*', 'occ_test.lst'))
# onet_test_paths.sort()

# onet_train_paths = glob.glob(join(onet_path, '*', 'occ_train.lst'))
# onet_train_paths.sort()

# onet_val_paths = glob.glob(join(onet_path, '*', 'occ_val.lst'))
# onet_val_paths.sort()


# actual_train_paths = glob.glob(join(actual_path, '*_train.lst'))
# actual_train_paths.sort()

# for idx in range(len(actual_test_paths)):
#     actual_test_path = actual_test_paths[idx]
#     expect_test_path = expect_test_paths[idx]
#     actual_train_path = actual_train_paths[idx]
#     r2n2_test_path = r2n2_test_paths[idx]
#     r2n2_train_path = r2n2_train_paths[idx]
#     onet_test_path = onet_test_paths[idx]
#     onet_train_path = onet_train_paths[idx]
#     onet_val_path = onet_val_paths[idx]

#     with open(actual_test_path, 'r') as f:
#         actual_lines = f.read().splitlines()
        
#     with open(expect_test_path, 'r') as f:
#         expect_lines = f.read().splitlines()

#     with open(actual_train_path, 'r') as f:
#         actual_train_lines = f.read().splitlines()
    
#     with open(r2n2_test_path, 'r') as f:
#         r2n2_test_lines = f.read().splitlines()

#     with open(r2n2_train_path, 'r') as f:
#         r2n2_train_lines = f.read().splitlines()

#     with open(onet_test_path, 'r') as f:
#         onet_test_lines = f.read().splitlines()
    
#     with open(onet_train_path, 'r') as f:
#         onet_train_lines = f.read().splitlines()
    
#     with open(onet_val_path, 'r') as f:
#         onet_val_lines = f.read().splitlines()

#     #print("Sorted", actual_lines == expect_lines)
#     # random.shuffle(expect_lines)
#     # print("Shuffled", actual_lines == expect_lines)

#     # overlap_list = []
#     # for test_item in actual_lines:
#     #     if test_item in actual_train_lines:
#     #         overlap_list.append(test_item)
#     # print(len(overlap_list))
    
#     r2n2_overlap_list = []
#     for test_item in actual_lines:
#         if test_item in r2n2_train_lines:
#             r2n2_overlap_list.append(test_item)

#     onet_other_lines = onet_train_lines + onet_val_lines
#     onet_test_lines.sort()
#     actual_lines.sort()
#     onet_overlap_list = []
#     onet_test_missing_list = []
#     for test_item in actual_lines:
#         if test_item in onet_other_lines:
#             onet_overlap_list.append(test_item)
#         if test_item not in onet_test_lines:
#             onet_test_missing_list.append(test_item)

    
#     print(f"Class:{actual_test_path.split('/')[-1].split('_')[0]}, Onet Overlap: {len(onet_overlap_list)}, Onet missing Acutal: {len(onet_test_missing_list)}, O: {len(onet_test_lines)}, A: {len(actual_lines)}, Occupancy Network and AtlasNet Test Equal: {onet_test_lines == actual_lines}")
    
    #print(f"R2N2 OverLap: {len(r2n2_overlap_list)}, R2N2 Check: {actual_lines == r2n2_test_lines}")


        
