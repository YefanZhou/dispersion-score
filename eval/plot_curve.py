import sys
sys.path.append("../")
from eval.eval_utils import mean_std
import matplotlib.pyplot as plt


pred_sscore_50=[0.269210, 0.275370, 0.279481, 0.272781, 0.268301, 0.266230, 0.269051, 0.267715, 0.272901]
pred_sscore_10=[0.279344, 0.279421, 0.266587, 0.260858, 0.269635, 0.268883, 0.267056, 0.272901, 0.261332]

gt_sscore_50 = [0.232200]
gt_sscore_10 = [0.240483]

##use gt label partition 
pred_sscore_50_opt0=[0.192917, 0.217128, 0.220873, 0.222589, 0.224312, 0.224860, 0.227613, 0.226210, 0.227114]
pred_sscore_10_opt0=[0.220065, 0.234886, 0.235671, 0.236941, 0.238880, 0.238656, 0.239459, 0.238639, 0.239894]

gt_sscore_50_opt0=[0.232200]
gt_sscore_10_opt0=[0.240483]

##use pred nviews=1 partition
pred_sscore_50_opt1=[0.269210, 0.228884, 0.218579, 0.215523, 0.211271, 0.209310, 0.209060, 0.203926, 0.205336]
pred_sscore_10_opt1=[0.279344, 0.249332, 0.239223, 0.238796, 0.233644, 0.231945, 0.230410, 0.228395, 0.228120]

gt_sscore_50_opt1=[0.157900]
gt_sscore_10_opt1=[0.182787]

##use pred nviews=23 partition
pred_sscore_50_opt2=[0.230353, 0.256304, 0.258840, 0.260238, 0.259554, 0.259163, 0.259698, 0.258200, 0.272901]
pred_sscore_10_opt2=[0.235511, 0.249504, 0.250534, 0.252499, 0.250733, 0.249799, 0.250899, 0.251260, 0.261332]

gt_sscore_50_opt2=[0.209247]
gt_sscore_10_opt2=[0.208293]


## fit data and predict 
lof_opt0 = [1.144425, 1.155395, 1.161543, 1.162079, 1.164996, 1.160966, 1.162429, 1.167469, 1.162593]
lof_opt1 = [1.083579, 1.089870, 1.096789, 1.096187, 1.101131, 1.099969, 1.102727, 1.104567, 1.102326]
gt_opt0=[1.180614]

## hsic 
values_mean = [63.358254, 60.877788, 60.075963, 60.416304, 59.826167, 59.800154, 59.281911, 59.432751, 59.714851]
values_std = [0.772557, 0.637112, 0.666286, 0.629291, 0.750259, 0.760181, 0.783761, 0.829284, 0.774827]
th_mean = [0.379584, 0.380623, 0.380799, 0.380228, 0.380796, 0.380836, 0.381212, 0.381077, 0.380571]
th_std = [0.000728, 0.000379, 0.000552, 0.00042, 0.000475, 0.000515, 0.000462, 0.000702, 0.000383]
gt_value_mean = [56.902183]
gt_value_std = [0.806149]
gt_th_mean = [0.383391]
gt_th_std = [0.000432]

##
train_pts_sscore_50 = [0.230377, 0.211950, 0.209952, 0.208687, 0.209003, 0.212863, 0.206239]
train_pts_sscore_10 = [0.236052, 0.234658, 0.221940, 0.225707, 0.231406, 0.227764, 0.227751]

train_imgs_sscore_50 = [0.012923, 0.015928, 0.019558, -0.000487, 0.014750, 0.011051, -0.001060]
train_imgs_sscore_10 = [0.006873, 0.012439, 0.029455, 0.021867, 0.010932, 0.000672, -0.002064]

chamfer_loss = [0.005405, 0.004117, 0.003711, 0.003554, 0.003400, 0.003332, 0.003275, 0.003202, 0.003201]
chamfer_loss_val = [0.000184, 0.000126, 0.000104, 0.000097, 0.000096, 0.000097, 0.000102, 0.000086, 0.000094]
fscore_loss = [0.518849, 0.583710, 0.608033, 0.621837, 0.633921, 0.636661, 0.645585, 0.649820, 0.650253]
fscore_loss_val = [0.003401, 0.003520, 0.002114, 0.003438, 0.002746, 0.003562, 0.002910, 0.003255, 0.003028]

train_nviews=[1, 3, 6, 9, 12, 15, 18, 21, 23]

# RGB GrayScale 
grayscale = {"fscore":[0.5373429, 0.5355308, 0.5358145],
"test_chamfer":[7.2588, 7.1571, 7.0554],
"train_chamer":[1.6709, 1.6632, 1.6823]}

rgb= {"fscore":[0.536445, 0.537323, 0.536711], 
"test_chamfer":[7.0349, 7.0361, 7.17],
"train_chamer":[1.6842, 1.6813, 1.6881]}

autoaug_cifar = {"fscore":[0.5218348, 0.5201172],
"test_chamfer":[7.5606, 7.337],
"train_chamer":[1.8525, 1.9776]}

autoaug_imagenet = {"fscore":[0.5216092, 0.5251945],
"test_chamfer":[7.4982, 7.5357],
"train_chamer":[1.8936, 1.8528]}

binary = {"fscore":[0.5252019, 0.5255882, 0.5247550],
"test_chamfer":[8.0428, 8.1135, 8.0844],
"train_chamer":[1.6862, 1.682, 1.6872]}


ap_10_mean = [0.93410, 0.95303, 1.28084, 2.07180, 2.79293]
ap_10_std = [0, 0, 0.17557, 0.23443, 0]
km_5_mean = [0.83398, 0.85535, 1.08647, 1.47646, 2.70000]
km_5_std = [0, 0, 0.13647, 0.16419, 0]


xticks = ["GrayScale", "RGB", "AutoAugment\n(Cifar10)", "AutoAugment\n(ImageNet)", "Binary"]
# plt.plot(xticks, ap_10_mean, label='AP_10')
# plt.fill_between(xticks, [ap_10_mean[i]-ap_10_std[i] for i in range(len(ap_10_mean))], [ap_10_mean[i]+ap_10_std[i] for i in range(len(ap_10_mean))], facecolor='gray', alpha=0.2)
# plt.plot(xticks, km_5_mean, label='KM_5')
# plt.fill_between(xticks, [km_5_mean[i]-km_5_std[i] for i in range(len(km_5_std))], [km_5_mean[i]+km_5_std[i] for i in range(len(km_5_std))], facecolor='gray', alpha=0.2)
# plt.legend()
# plt.ylabel("Inertia", fontsize=10)
# plt.xlabel("Augment Method", fontsize=10)
# plt.title("Inertia Score vs image augment method")
# plt.show()


# fscore_mean = [mean_std(grayscale['fscore'])[0], mean_std(rgb['fscore'])[0], 
#              mean_std(autoaug_cifar['fscore'])[0],
#             mean_std(autoaug_imagenet['fscore'])[0], mean_std(binary['fscore'])[0]]
# fscore_std = [mean_std(grayscale['fscore'])[1], mean_std(rgb['fscore'])[1], 
#             mean_std(autoaug_cifar['fscore'])[1],
#             mean_std(autoaug_imagenet['fscore'])[1], mean_std(binary['fscore'])[1]]

# test_cd_mean = [mean_std(grayscale['test_chamfer'])[0], mean_std(rgb['test_chamfer'])[0], 
#              mean_std(autoaug_cifar['test_chamfer'])[0],
#             mean_std(autoaug_imagenet['test_chamfer'])[0], mean_std(binary['test_chamfer'])[0]]
# test_cd_std = [mean_std(grayscale['test_chamfer'])[1], mean_std(rgb['test_chamfer'])[1], 
#              mean_std(autoaug_cifar['test_chamfer'])[1],
#             mean_std(autoaug_imagenet['test_chamfer'])[1], mean_std(binary['test_chamfer'])[1]]

# train_cd_mean = [mean_std(grayscale['train_chamer'])[0], mean_std(rgb['train_chamer'])[0], 
#              mean_std(autoaug_cifar['train_chamer'])[0],
#             mean_std(autoaug_imagenet['train_chamer'])[0], mean_std(binary['train_chamer'])[0]]

# train_cd_std = [mean_std(grayscale['train_chamer'])[1], mean_std(rgb['train_chamer'])[1], 
#              mean_std(autoaug_cifar['train_chamer'])[1],
#             mean_std(autoaug_imagenet['train_chamer'])[1], mean_std(binary['train_chamer'])[1]]


## 


# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(xticks, fscore_mean, 'o-')
# plt.fill_between(xticks, [fscore_mean[i]-fscore_std[i] for i in range(len(fscore_mean))], [fscore_mean[i]+fscore_std[i] for i in range(len(fscore_mean))], facecolor='gray', alpha=0.2)
# plt.ylabel("F-score", fontsize=10)
# plt.title("F-score")
# plt.subplot(3, 1, 2)
# plt.plot(xticks, test_cd_mean, 'o-')
# plt.fill_between(xticks, [test_cd_mean[i]-test_cd_std[i] for i in range(len(test_cd_mean))], [test_cd_mean[i]+test_cd_std[i] for i in range(len(test_cd_mean))], facecolor='gray', alpha=0.2)
# plt.ylabel("Test Chamfer", fontsize=10)
# plt.title("Test Chamfer(1e-3)")
# plt.subplot(3, 1, 3)
# plt.plot(xticks, train_cd_mean, 'o-')
# plt.fill_between(xticks, [train_cd_mean[i]-train_cd_std[i] for i in range(len(train_cd_mean))], [train_cd_mean[i]+train_cd_std[i] for i in range(len(train_cd_mean))], facecolor='gray', alpha=0.2)
# plt.ylabel("Train Chamfer", fontsize=10)
# plt.title("Train Chamfer(1e-3)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# factor = [0.1, 0.3, 0.5, 0.7, 1]
# #b_f_score = [0.5364, 0.5378, 0.5385, 0.5392, mean_std(rgb['fscore'])[0]]
# #b_f_score = [0.514, 0.5183, 0.5275, 0.5359, mean_std(rgb['fscore'])[0]]
# b_f_score = [0.5378, 0.5388, 0.5396, 0.5404, mean_std(rgb['fscore'])[0]]
# b_f_score_std = [0, 0, 0, 0, mean_std(rgb['fscore'])[1]]

# #b_test_chamfer = [6.9727, 7.092, 7.0202, 7.1348, mean_std(rgb['test_chamfer'])[0]]
# #b_test_chamfer = [7.7506, 7.6695, 7.448, 7.177, mean_std(rgb['test_chamfer'])[0]]
# b_test_chamfer = [7.2078, 7.1257, 6.9059, 6.9381, mean_std(rgb['test_chamfer'])[0]]
# b_test_chamfer_std = [0, 0, 0, 0, mean_std(rgb['test_chamfer'])[1]]

# #b_train_chamfer = [1.7312, 1.656, 1.713, 1.672, mean_std(rgb['train_chamer'])[0]]
# #b_train_chamfer = [1.8462, 1.8867, 1.7139, 1.6767, mean_std(rgb['train_chamer'])[0]]
# b_train_chamfer = [1.676, 1.6709, 1.7392, 1.6646, mean_std(rgb['train_chamer'])[0]]
# b_train_chamfer_std = [0, 0, 0, 0, mean_std(rgb['train_chamer'])[1]]

# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(factor, b_f_score, 'o-')
# plt.fill_between(factor, [b_f_score[i]-b_f_score_std[i] for i in range(len(b_f_score))], [b_f_score[i]+b_f_score_std[i] for i in range(len(b_f_score))], facecolor='gray', alpha=0.2)
# plt.xticks(factor)
# plt.ylabel("F-score", fontsize=10)
# plt.title("F-score")
# plt.subplot(3, 1, 2)
# plt.plot(factor, b_test_chamfer, 'o-')
# plt.fill_between(factor, [b_test_chamfer[i]-b_test_chamfer_std[i] for i in range(len(b_test_chamfer))], [b_test_chamfer[i]+b_test_chamfer_std[i] for i in range(len(test_cd_mean))], facecolor='gray', alpha=0.2)
# plt.ylabel("Test Chamfer", fontsize=10)
# plt.xticks(factor)
# plt.title("Test Chamfer(1e-3)")
# plt.subplot(3, 1, 3)
# plt.plot(factor, b_train_chamfer, 'o-')
# plt.fill_between(factor, [b_train_chamfer[i]-b_train_chamfer_std[i] for i in range(len(b_train_chamfer))], [b_train_chamfer[i]+b_train_chamfer_std[i] for i in range(len(b_train_chamfer))], facecolor='gray', alpha=0.2)
# plt.ylabel("Train Chamfer", fontsize=10)
# plt.title("Train Chamfer(1e-3)")
# plt.xticks(factor)
# plt.legend()
# plt.tight_layout()
# plt.show()



# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_nviews, pred_sscore_50, "-o")
# plt.plot(train_nviews, len(train_nviews) * gt_sscore_50)
# plt.ylim((0.230, 0.300))
# plt.legend(["Pred SScore", "GT SScore"])
# plt.title("Pred PointCloud SScore \nperference percentile 50%")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("Silhouette Score")

# plt.subplot(1, 2, 2)
# plt.plot(train_nviews, pred_sscore_10, "-o")
# plt.plot(train_nviews, len(train_nviews) * gt_sscore_10)
# plt.ylim((0.230, 0.300))
# plt.legend(["Pred SScore", "GT SScore"])
# plt.title("Pred PointCloud SScore \nperference percentile 10%")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("Silhouette Score")
# plt.show()


# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_nviews, train_pts_sscore_50, "-o")
# #plt.plot(train_nviews, len(train_nviews) * gt_sscore_50)
# plt.title("Train PointCloud SScore \nperference percentile 50%")
# plt.ylim((0.190, 0.250))
# plt.xticks(train_nviews)

# plt.subplot(1, 2, 2)
# plt.plot(train_nviews, train_pts_sscore_10, "-o")
# plt.ylim((0.190, 0.250))
# #plt.plot(train_nviews, len(train_nviews) * gt_sscore_10)
# plt.title("Train PointCloud SScore \nperference percentile 10%")
# plt.xticks(train_nviews)
# plt.show()


# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_nviews, train_imgs_sscore_50, "-o")
# #plt.plot(train_nviews, len(train_nviews) * gt_sscore_50)
# plt.title("Train Image SScore \nperference percentile 50%")
# plt.ylim((-0.010, 0.030))
# plt.xticks(train_nviews)

# plt.subplot(1, 2, 2)
# plt.plot(train_nviews, train_imgs_sscore_10, "-o")
# plt.ylim((-0.010, 0.030))
# #plt.plot(train_nviews, len(train_nviews) * gt_sscore_10)
# plt.title("Train Image SScore \nperference percentile 10%")
# plt.xticks(train_nviews)
# plt.show()


# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_nviews, chamfer_loss, "-o")
# plt.fill_between(train_nviews, [chamfer_loss[i] - chamfer_loss_val[i] for i in range(len(chamfer_loss))], [chamfer_loss[i] + chamfer_loss_val[i] for i in range(len(chamfer_loss))], facecolor='gray', alpha=0.2)
# plt.title("Chamfer Loss")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")

# plt.subplot(1, 2, 2)
# plt.plot(train_nviews, fscore_loss, "-o")
# plt.fill_between(train_nviews, [fscore_loss[i] - fscore_loss_val[i] for i in range(len(fscore_loss))], [fscore_loss[i] + fscore_loss_val[i] for i in range(len(fscore_loss))], facecolor='gray', alpha=0.2)
# plt.title("Fscore @ 0.001")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
#plt.show()
#plt.savefig('scripts/shapenet13_pred/Chamfer_Fscore.png')

#####use gt label partition 
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_nviews, pred_sscore_50_opt0, "-o")
# plt.plot(train_nviews, len(train_nviews) * gt_sscore_50_opt0)
# plt.legend(["Pred SScore", "GT SScore"])
# plt.ylim((0.190, 0.250))
# plt.title("Pred PointCloud SScore \nperference percentile 50%")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("Silhouette Score")

# plt.subplot(1, 2, 2)
# plt.plot(train_nviews, pred_sscore_10_opt0, "-o")
# plt.plot(train_nviews, len(train_nviews) * gt_sscore_10_opt0)
# plt.legend(["Pred SScore", "GT SScore"])
# plt.ylim((0.190, 0.250))
# plt.title("Pred PointCloud SScore \nperference percentile 10%")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("Silhouette Score")
# plt.show()

####use pred nviews=1 partition
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_nviews, pred_sscore_50_opt1, "-o")
# plt.plot(train_nviews, len(train_nviews) * gt_sscore_50_opt1)
# plt.legend(["Pred SScore", "GT SScore"])
# plt.ylim((0.150, 0.285))
# plt.title("Pred PointCloud SScore \nperference percentile 50%")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("Silhouette Score")

# plt.subplot(1, 2, 2)
# plt.plot(train_nviews, pred_sscore_10_opt1, "-o")
# plt.plot(train_nviews, len(train_nviews) * gt_sscore_10_opt1)
# plt.legend(["Pred SScore", "GT SScore"])
# plt.ylim((0.150, 0.285))
# plt.title("Pred PointCloud SScore \nperference percentile 10%")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("Silhouette Score")
# plt.show()


####use pred nviews=23 partition
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_nviews, pred_sscore_50_opt2, "-o")
# plt.plot(train_nviews, len(train_nviews) * gt_sscore_50_opt2)
# plt.legend(["Pred SScore", "GT SScore"])
# #plt.ylim((0.150, 0.285))
# plt.title("Pred PointCloud SScore \nperference percentile 50%")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("Silhouette Score")

# plt.subplot(1, 2, 2)
# plt.plot(train_nviews, pred_sscore_10_opt2, "-o")
# plt.plot(train_nviews, len(train_nviews) * gt_sscore_10_opt2)
# plt.legend(["Pred SScore", "GT SScore"])
# #plt.ylim((0.150, 0.285))
# plt.title("Pred PointCloud SScore \nperference percentile 10%")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("Silhouette Score")
# plt.show()

##LOF
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_nviews, lof_opt0, "-o")
# plt.plot(train_nviews, len(train_nviews) * gt_opt0)
# plt.legend(["Pred LOF", "GT LOF"])
# #plt.ylim((0.150, 0.285))
# plt.title("Pred PointCloud LOF, Model Fit Each Pred Set")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("LOF")

# plt.subplot(1, 2, 2)
# plt.plot(train_nviews, lof_opt1, "-o")
# #plt.plot(train_nviews, len(train_nviews) * gt_sscore_10_opt2)
# #plt.legend(["Pred SScore", "GT SScore"])
# #plt.ylim((0.150, 0.285))
# plt.title("Pred PointCloud LOF, Model Fit GT, Detect Pred")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("LOF")
# plt.show()

##HSIC 
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_nviews, values_mean, "-o")
# plt.plot(train_nviews, len(train_nviews) * gt_value_mean)
# plt.legend(["Pred HSIC", "GT HSIC"])
# #plt.ylim((0.150, 0.285))
# plt.title("Pred PointCloud HSIC Value")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("HSIC value")

# plt.subplot(1, 2, 2)
# plt.plot(train_nviews, th_mean, "-o")
# plt.plot(train_nviews, len(train_nviews) * gt_th_mean)
# plt.legend(["Pred HSIC", "GT HSIC"])
# #plt.ylim((0.150, 0.285))
# plt.title("Pred PointCloud HSIC Threshold")
# plt.xticks(train_nviews)
# plt.xlabel("Num of views per shape in Train Set")
# plt.ylabel("HSIC")
#plt.show()


