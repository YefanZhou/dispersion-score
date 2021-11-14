import torch
import torch.optim as optim
import auxiliary.my_utils as my_utils
import json
from os.path import join, exists
import auxiliary.meter as meter
import time


class TrainerAbstract(object):
    """
    This class implements an abtsract deep learning trainer. It is supposed to be generic for any data, task, architecture, loss...
    It defines the usual generic fonctions.
    
    """

    def __init__(self, opt):
        super(TrainerAbstract, self).__init__()
        self.start_time = time.time()
        self.opt = opt
        self.get_log_paths()
        self.init_meters()
        self.reset_epoch()
        if not opt.demo:
            my_utils.print_arg(self.opt)

    def get_log_paths(self):
        """
        Define paths to save and reload networks from parsed options
        :return:
        """

        # if not self.opt.demo:
        #     if not exists("log"):
        #         self.opt.logger.info("Creating log folder")
        #         mkdir("log")
        #     if not exists(self.opt.dir_name):
        #         self.opt.logger.info("creating folder  ", self.opt.dir_name)
        #         mkdir(self.opt.dir_name)

        self.opt.log_path = join(self.opt.dir_name, "log.txt")
        self.opt.optimizer_path = join(self.opt.dir_name, 'optimizer.pth')
        self.opt.model_path = join(self.opt.dir_name, "network.pth")
        self.opt.reload_optimizer_path = ""

        # # If a network is already created in the directory
        if exists(self.opt.model_path):
            self.opt.reload_model_path = self.opt.model_path
            self.opt.reload_optimizer_path = self.opt.optimizer_path

    def init_meters(self):
        self.log = meter.Logs()

    def print_loss_info(self):
        pass

    def save_network(self):
        self.opt.logger.info("saving net...")
        # if self.epoch >= self.opt.nepoch - 2 and self.epoch != self.opt.nepoch:
        #     last_model_path = join(self.opt.dir_name, f"network_epoch{self.epoch}.pth")
        #     torch.save(self.network.state_dict(), last_model_path)
        
        if self.log.best_save_flag:
            self.opt.logger.info(f"saving best net, best {self.log.model_select_metric}: {self.log.best_model_select_metric}")
            best_model_path = join(self.opt.dir_name, f"network_best.pth")
            torch.save(self.network.state_dict(), best_model_path)

        torch.save(self.network.state_dict(), self.opt.model_path)
        #torch.save(self.optimizer.state_dict(), self.opt.optimizer_path)
        self.opt.logger.info("network saved")

    def dump_stats(self):
        """
        Save stats at each epoch
        """

        log_table = {
            "epoch": self.epoch + 1,
            "lr": self.opt.lrate,
            "env": self.opt.dir_name,
        }
        log_table.update(self.log.current_epoch)

        with open(self.opt.log_path, "a") as f:  # open and append
            f.write("json_stats: " + json.dumps(log_table) + "\n")
        
        self.opt.logger.info(log_table)

        self.opt.start_epoch = self.epoch
        with open(join(self.opt.dir_name, "options.json"), "w") as f:  # open and append
            save_dict = dict(self.opt.__dict__)
            save_dict.pop("device")
            save_dict.pop("tbwriter")
            save_dict.pop("logger")
            f.write(json.dumps(save_dict))

    def print_iteration_stats(self, loss):
        """
        print stats at each iteration
        """
        current_time = time.time()
        ellpased_time = current_time - self.start_train_time
        total_time_estimated = self.opt.nepoch * (self.datasets.len_dataset / self.opt.batch_size) * ellpased_time / (
                0.00001 + self.iteration + 1.0 * self.epoch * self.datasets.len_dataset / self.opt.batch_size)  # regle de 3
        ETL = total_time_estimated - ellpased_time
        self.opt.logger.info(
            f"["
            + f"{self.epoch}"
            + f": "
            + f"{self.iteration}"
            + "/"
            + f"{int(self.datasets.len_dataset / self.opt.batch_size)}"
            + "] chamfer train loss:  "
            + f"{loss.item() :5f} "
            + f"Ellapsed Time: {ellpased_time / 60 / 60 :3f}h "
            + f"ETL: {ETL / 60 / 60 :3f}h")
        

    def learning_rate_scheduler(self):
        """
        Defines the learning rate schedule
        """
        # Warm-up following https://arxiv.org/pdf/1706.02677.pdf
        # if len(self.next_learning_rates) > 0:
        #     next_learning_rate = self.next_learning_rates.pop()
        #     self.opt.logger.info(f"warm-up learning rate {next_learning_rate}")
        #     for g in self.optimizer.param_groups:
        #         g['lr'] = next_learning_rate

        # Learning rate decay
        if self.epoch == self.opt.lr_decay_1:
            self.opt.lrate = self.opt.lrate / 10.0
            self.opt.logger.info(f"First learning rate decay {self.opt.lrate}")
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if self.epoch == self.opt.lr_decay_2:
            self.opt.lrate = self.opt.lrate / 10.0
            self.opt.logger.info(f"Second learning rate decay {self.opt.lrate}")
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if self.epoch == self.opt.lr_decay_3:
            self.opt.lrate = self.opt.lrate / 10.0
            self.opt.logger.info(f"Third learning rate decay {self.opt.lrate}")
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)

    def increment_epoch(self):
        self.epoch = self.epoch + 1

    def increment_iteration(self):
        self.iteration = self.iteration + 1

    def reset_iteration(self):
        self.iteration = 0

    def reset_epoch(self):
        self.epoch = self.opt.start_epoch
