#Generate Train Test Split List 
# logging.basicConfig(level=logging.INFO)
# opt = {"normalization": "UnitBall", "class_choice": ["plane"], "SVR": True, "sample": True, "npoints": 2500,
#        "shapenet13": True, "demo": False, "logger": logging.getLogger(), "img_aug": False, 
#        "img_aug_type": 'rgb', "color_aug_factor":[0, 1, 1, 0],  "manual_seed": 1, "no_compile_chamfer": False,
#        "autoaug_type": 'RGB', "number_points":2500, "mag_idx":0, "prob":0.0, "n_op":0, "test_augment": False}
# opt = EasyDict(opt)
# my_utils.plant_seeds(opt.manual_seed)
# if_train = True
# dataset = ShapeNet(opt, train=if_train, num_image_per_object=1)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=8,
#                                         shuffle=False,
#                                         num_workers=16)


# train_test_dic = {}
# total_list = []

# for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
#     cats = batch['category']
#     f_names = batch['name']
#     for cat, f_name in zip(cats, f_names):
#         if cat not in train_test_dic:
#             train_test_dic[cat] = []
#         train_test_dic[cat].append(f_name.split('.')[0])
#         total_list.append(join(cat, f_name.split('.')[0]))

# total_count = 0
# # for cat in train_test_dic:
# #     if if_train:
# #         fo = open(f"data/filelists/{cat}_train.lst", "w")
# #     else:
# #         fo = open(f"data/filelists/{cat}_test.lst", "w")
# #     total_count += len(train_test_dic[cat])
# #     for f_id in train_test_dic[cat]:
# #         fo.write(f_id + '\n')

# if if_train:
#     fo = open(f"data/filelists/train.lst", "w")
# else:
#     fo = open(f"data/filelists/test.lst", "w")
# for f_id in total_list:
#     fo.write(f_id + '\n')

# print(len(total_list))