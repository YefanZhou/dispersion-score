from termcolor import colored
import numpy as np
from os.path import join
import cv2
import torchvision.transforms as transforms

class TrainerIteration(object):
    """
        This class implements all functions related to a single forward pass of Atlasnet.
    """

    def __init__(self):
        super(TrainerIteration, self).__init__()

    def make_network_input(self):
        """
        Arrange to data to be fed to the network.
        :return:
        """
        if self.opt.SVR:
            self.data.network_input = self.data.image.to(self.opt.device)
        else:
            self.data.network_input = self.data.points.transpose(2, 1).contiguous().to(self.opt.device)

    def common_ops(self):
        """
        Commom operations between train and eval forward passes
        :return:
        """
        self.make_network_input()
        self.batch_size = self.data.points.size(0)

        self.data.pointsReconstructed_prims = self.network(self.data.network_input,
                                                           train=self.flags.train)
        
        self.fuse_primitives()

        self.loss_model()  # batch

        if not self.flags.train and self.opt.run_single_eval and self.opt.save_pred:
            self.save_prediction()

        #self.visualize()

    def train_iteration(self):
        """
        Forward backward pass
        :return:
        """
        self.optimizer.zero_grad()
        self.common_ops()
        self.log.update("loss_train_total", self.data.loss.item(), self.opt.logger)
        self.opt.tbwriter.add_scalar("Train/loss_train_total_iter", self.data.loss.item(), self.cum_iteration)
        if not self.opt.no_learning:
            self.data.loss.backward()
            self.optimizer.step()  # gradient update
        if self.iteration % 100 == 0:
            self.print_iteration_stats(self.data.loss)

    def visualize(self):
        if self.iteration % 50 == 1:
            tmp_string = "train" if self.flags.train else "test"
            self.visualizer.show_pointcloud(self.data.points[0], title=f"GT {tmp_string}")
            self.visualizer.show_pointcloud(self.data.pointsReconstructed[0], title=f"Reconstruction {tmp_string}")
            if self.opt.SVR:
                self.visualizer.show_image(self.data.image[0], title=f"Input Image {tmp_string}")
    
    def save_prediction(self):
        pred_path = join(self.opt.results_dir, 
                f"pred_eval_bs{self.opt.batch_size_test}_iter{self.iteration}_idx0.npy")
        gt_path = join(self.opt.results_dir, 
                f"gt_eval_bs{self.opt.batch_size_test}_iter{self.iteration}_idx0.npy")
        img_path = join(self.opt.results_dir, 
                f"img_eval_bs{self.opt.batch_size_test}_iter{self.iteration}_idx0.png")

        np.save(pred_path, self.data.pointsReconstructed[0].cpu().numpy())
        np.save(gt_path, self.data.points[0].cpu().numpy())
        imgPIL = transforms.ToPILImage()(self.data.image[0])
        imgPIL.save(img_path)
        #img_tosave = np.transpose(self.data.image[0].cpu().numpy(), (1, 2, 0))
        #cv2.imwrite(img_path, img_tosave)
 



    def test_iteration(self):
        """
        Forward evaluation pass
        :return:
        """
        self.common_ops()
        self.num_val_points = self.data.pointsReconstructed.size(1)
        self.log.update("loss_val", self.data.loss.item(), self.opt.logger)
        self.log.update("fscore", self.data.loss_fscore.item(), self.opt.logger)
        self.opt.logger.info(
                '[%d: %d/%d]' % (self.epoch, self.iteration, self.datasets.len_dataset_test / self.opt.batch_size_test) +
            'loss_val:  %f' % self.data.loss.item())
