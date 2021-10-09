import numpy as np
import torch
from dataset.toy_dataset.toydataset import ToyDataset
from model.model import EncoderDecoder
import argparse
from auxiliary.argument_parser import parser
import auxiliary.my_utils as my_utils
import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
from auxiliary.ChamferDistancePytorch.fscore import fscore
import torch.optim as optim
from os.path import join
import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    opt = parser()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    my_utils.plant_seeds(manual_seed=opt.manual_seed)
    network = EncoderDecoder(opt)

    if opt.run_single_eval:
        opt.logger.info(f"Reloading Network Weights from {opt.reload_model_path}...")
        network.load_state_dict(torch.load(opt.reload_model_path)['model_state_dict'])
    
    network.to(opt.device)
    test_dataset = ToyDataset(opt.data_base_dir, 
        json_file=opt.test_json_file, 
        num_points=opt.number_points, 
        train=False,
        normalization=opt.normalization, 
        logger=opt.logger)
    opt.logger.info(f"Test Dataset Size: {len(test_dataset)}")

    test_loader = \
        torch.utils.data.DataLoader(test_dataset, 
                            batch_size=opt.batch_size_test, 
                            shuffle=False, 
                            num_workers=opt.workers)

    optimizer = optim.Adam(network.parameters(), lr=opt.lrate)

    if not opt.run_single_eval:
        train_dataset = ToyDataset(opt.data_base_dir, 
                    json_file=opt.train_json_file, 
                    num_points=opt.number_points, 
                    train=True,
                    normalization=opt.normalization, 
                    logger=opt.logger)
        val_dataset = ToyDataset(opt.data_base_dir, 
                    json_file=opt.val_json_file, 
                    num_points=opt.number_points,
                    train=False, 
                    normalization=opt.normalization, 
                    logger=opt.logger)

    
        opt.logger.info(f"Train Dataset Size: {len(train_dataset)}, Val Dataset Size: {len(val_dataset)}")

        train_loader = \
            torch.utils.data.DataLoader(train_dataset, 
                                    batch_size=opt.batch_size, 
                                    shuffle=True, 
                                    num_workers=opt.workers)

        val_loader = \
            torch.utils.data.DataLoader(val_dataset, 
                                    batch_size=opt.batch_size_test, 
                                    shuffle=False, 
                                    num_workers=opt.workers)

    trainer = Trainer(opt, optimizer, network)

    if not opt.run_single_eval:
        best_loss = float('inf')
        start_time = time.time()
        for epoch in range(1, opt.nepoch):
            # Train
            train_loss = trainer.train(train_loader)
            opt.logger.info(f"Train Epoch {epoch}/{opt.nepoch}: chamfer loss: {train_loss:.4f}")
            opt.tbwriter.add_scalar('Train/chamfer_epoch', train_loss, epoch)
            # Val
            val_loss, val_fscore = trainer.test(val_loader, vis=False)

            trainer.learning_rate_scheduler(epoch)

            opt.logger.info(f"Val Epoch {epoch}/{opt.nepoch}: chamfer loss: {val_loss:.4f}, fscore: {val_fscore:.4f}")
            opt.tbwriter.add_scalar('Val/chamfer_epoch', val_loss, epoch)
            opt.tbwriter.add_scalar('Val/fscore_epoch', val_fscore, epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                opt.logger.info(f"Train Epoch {epoch}/{opt.nepoch}: chamfer loss: {train_loss:.4f}")
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': trainer.network.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'loss': best_loss,
                        }, join(opt.dir_name, 'best_checkpoint.pt'))
                opt.logger.info(f"val new best loss:{best_loss:.4f}, saving checkpoints...")

            if epoch % 100 == 0 or epoch == opt.nepoch - 1:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': trainer.network.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'loss': val_loss,
                        }, join(opt.dir_name, 'checkpoint.pt'))
                opt.logger.info(f"backup saving loss:{val_loss:.4f}, saving checkpoints...")

                pass_time = (time.time() - start_time) / 60 
                remain_time = (pass_time / epoch) * (opt.nepoch - epoch)
                opt.logger.info(f"Passed Time:{pass_time} min, Remained Time: {remain_time} min")
                

        test_loss, test_fscore, vis_ptclouds = trainer.test(test_loader, vis=True)
        opt.logger.info(f"Test Epoch {epoch}/{opt.nepoch}: chamfer loss: {test_loss:.4f}, fscore: {test_fscore:.4f}")
        vis_ptclouds = vis_ptclouds.numpy()
        
        opt.logger.info(f"Saving prediction {vis_ptclouds.shape} to {join(opt.dir_name, 'prediction.npy')}")
        np.save(join(opt.dir_name, 'prediction.npy'), vis_ptclouds)
    
    else:
        test_loss, test_fscore, vis_ptclouds = trainer.test(test_loader, vis=True)
        opt.logger.info(f"Eval: chamfer loss: {test_loss:.4f}, fscore: {test_fscore:.4f}")
        vis_ptclouds = vis_ptclouds.numpy()
        opt.logger.info(f"Saving prediction {vis_ptclouds.shape} to {join(opt.results_dir, 'prediction_best.npy')}")
        np.save(join(opt.results_dir, 'prediction_best.npy'), vis_ptclouds)


class Trainer(object):
    def __init__(self, opt, optimizer, network):
        self.distChamfer = dist_chamfer_3D.chamfer_3DDist()
        self.opt = opt
        self.train_iter = 0
        self.device = opt.device
        self.tbwriter = opt.tbwriter
        self.optimizer = optimizer
        self.network = network

    def train(self, train_loader):
        self.network.train()
        loss_total = 0
        for batch in train_loader:
            image, ptcloud = batch['image'], batch['points']
            batch_size = image.shape[0]
            image = image.to(self.device)
            ptcloud = ptcloud.to(self.device)
            self.optimizer.zero_grad()
            pointsReconstructed_prims = self.network(image)
            pointsReconstructed = pointsReconstructed_prims.transpose(2, 3).contiguous()
            pointsReconstructed = pointsReconstructed.view(batch_size, -1, 3)

            loss = self.chamfer_loss(ptcloud, pointsReconstructed, train=True)
            loss.backward()
            self.optimizer.step()

            loss = loss.item() 
            self.train_iter += 1
            loss_total += loss
            self.tbwriter.add_scalar('Train/chamfer_iter', loss, self.train_iter)

        loss_avg = loss_total / len(train_loader)

        return loss_avg
    
    def test(self, loader, vis):
        self.network.eval()
        loss_total = 0
        loss_fscore = 0
        if vis:
            vis_list = []
        with torch.no_grad():
            for batch in loader:
                image, ptcloud = batch['image'], batch['points']
                batch_size = image.shape[0]
                image = image.to(self.device)
                ptcloud = ptcloud.to(self.device)
                pointsReconstructed_prims = self.network(image)
                pointsReconstructed = pointsReconstructed_prims.transpose(2, 3).contiguous()
                pointsReconstructed = pointsReconstructed.view(batch_size, -1, 3)
                loss, fscore = self.chamfer_loss(ptcloud, pointsReconstructed, train=False)
                
                loss = loss.item()
                fscore = fscore.item()
                loss_total += loss
                loss_fscore += fscore
                if vis:
                    vis_list.append(pointsReconstructed.detach().cpu())
        
        loss_avg = loss_total / len(loader)
        loss_fscore_avg = loss_fscore / len(loader)

        if vis:
            vis_ptclouds = torch.cat(vis_list, dim=0)
            return loss_avg, loss_fscore_avg, vis_ptclouds

        return loss_avg, loss_fscore_avg


    def chamfer_loss(self, points, pointsReconstructed, train=True):
        """
        Training loss of Atlasnet. The Chamfer Distance. Compute the f-score in eval mode.
        :return:
        """
        inCham1 = points.view(points.size(0), -1, 3).contiguous()
        inCham2 = pointsReconstructed.contiguous().view(points.size(0), -1, 3).contiguous()
        dist1, dist2, idx1, idx2 = self.distChamfer(inCham1, inCham2)  # mean over points
        loss = torch.mean(dist1) + torch.mean(dist2)  # mean over points
        
        if not train:
            loss_fscore, _, _ = fscore(dist1, dist2)
            loss_fscore = loss_fscore.mean()
            return loss, loss_fscore
        
        return loss


    def learning_rate_scheduler(self, epoch):
        if epoch == self.opt.lr_decay_1:
            self.opt.lrate = self.opt.lrate / 10.0
            self.opt.logger.info(f"First learning rate decay {self.opt.lrate}")
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if epoch == self.opt.lr_decay_2:
            self.opt.lrate = self.opt.lrate / 10.0
            self.opt.logger.info(f"Second learning rate decay {self.opt.lrate}")
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if epoch == self.opt.lr_decay_3:
            self.opt.lrate = self.opt.lrate / 10.0
            self.opt.logger.info(f"Third learning rate decay {self.opt.lrate}")
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)

if __name__ == "__main__":
    main()
