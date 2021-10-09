import torch
import dataset.dataset_shapenet as dataset_shapenet
import dataset.dataset_shapenet_views as dataset_shapenet_views
import dataset.dataset_shapenet_views_yawlimit as dataset_shapenet_views_yawlimit
import dataset.augmenter as augmenter
from easydict import EasyDict


class TrainerDataset(object):
    def __init__(self):
        super(TrainerDataset, self).__init__()

    def build_dataset(self):
        """
        Create dataset
        """

        self.datasets = EasyDict()
        # Create Datasets
        #assert self.opt.views_search != self.opt.random_view_select, "views_search, random_view_select should not both be enabled or disabled"
        print(f"views_search{self.opt.views_search} random_view_select{self.opt.random_view_select} anglelimit{self.opt.anglelimit_search}")
        if self.opt.views_search:
            self.opt.logger.info(f"DataSet Mode: views_search, Train {self.opt.nviews_train} of views per shape")
            if not self.opt.run_single_eval:
                self.datasets.dataset_train = \
                    dataset_shapenet_views.ShapeNet(self.opt, train=True, 
                                num_image_per_object=self.opt.nviews_train)
            self.datasets.dataset_test = \
                dataset_shapenet_views.ShapeNet(self.opt, train=False, 
                            num_image_per_object=self.opt.nviews_test)
        elif self.opt.random_view_select:
            self.opt.logger.info(f"DataSet Mode: random_view_select every epoch")
            self.datasets.dataset_train = \
                    dataset_shapenet.ShapeNet(self.opt, train=True)
            self.datasets.dataset_test = \
                    dataset_shapenet.ShapeNet(self.opt, train=False)
        elif self.opt.anglelimit_search:
            self.opt.logger.info(f"DataSet Mode: anglelimit_search: yawrange:{self.opt.yawrange} rendering root dir:{self.opt.rendering_root_dir}")
            self.datasets.dataset_train = \
                    dataset_shapenet_views_yawlimit.ShapeNet(self.opt, train=True)
            self.datasets.dataset_test = \
                    dataset_shapenet_views_yawlimit.ShapeNet(self.opt, train=False)
        else:
            raise NotImplementedError()
            # if not self.opt.run_single_eval:
            #     self.datasets.dataset_train = dataset_shapenet.ShapeNet(self.opt, train=True)
            # self.datasets.dataset_test = dataset_shapenet.ShapeNet(self.opt, train=False)

        if not self.opt.demo:
            # Create dataloaders
            if not self.opt.run_single_eval:
                self.datasets.dataloader_train = torch.utils.data.DataLoader(self.datasets.dataset_train,
                                                                         batch_size=self.opt.batch_size,
                                                                         shuffle=True,
                                                                         num_workers=int(self.opt.workers), pin_memory=True, drop_last=True)
            self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
                                                                        batch_size=self.opt.batch_size_test,
                                                                        shuffle=False, num_workers=int(self.opt.workers), pin_memory=True)
            axis = []
            if self.opt.data_augmentation_axis_rotation:
                axis = [1]

            flips = []
            if self.opt.data_augmentation_random_flips:
                flips = [0, 2]

            # Create Data Augmentation
            self.datasets.data_augmenter = augmenter.Augmenter(translation=self.opt.random_translation,
                                                               rotation_axis=axis,
                                                               anisotropic_scaling=self.opt.anisotropic_scaling,
                                                               rotation_3D=self.opt.random_rotation,
                                                               flips=flips)

            if not self.opt.run_single_eval:
                self.datasets.len_dataset = len(self.datasets.dataset_train)
            self.datasets.len_dataset_test = len(self.datasets.dataset_test)
