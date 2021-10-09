import torch
#from auxiliary.my_utils import weights_init
from model.model import EncoderDecoder
from model.baseline_models import SVR_Baseline, SVR_AtlasNet, SVR_AtlasNet_SPHERE, PSGN, FoldNet
import torch.optim as optim
import numpy as np
import torch.nn as nn
from copy import deepcopy
import os

class TrainerModel(object):
    def __init__(self):
        """
        This class creates the architectures and implements all trainer functions related to architecture.
        """
        super(TrainerModel, self).__init__()

    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        if torch.cuda.is_available():
            if self.opt.parallel:
                self.opt.device = torch.device(f"cuda:{self.opt.multi_gpu[0]}")
            else:
                self.opt.device = torch.device("cuda")
        else:
            # Run on CPU
            self.opt.device = torch.device(f"cpu")
        
        if self.opt.network == 'psgn':
            self.network = PSGN(num_points=self.opt.number_points, 
                                bottleneck_size=self.opt.bottleneck_size, 
                                hidden_neurons=self.opt.hidden_neurons, 
                                pretrained_encoder=False,
                                remove_all_batchNorms=self.opt.remove_all_batchNorms)
            
            self.network.to(self.opt.device)
            
        elif self.opt.network == 'atlasnet':
            self.network = EncoderDecoder(self.opt)
            ## AtlasNet implementation v2, https://github.com/ThibaultGROUEIX/AtlasNet/tree/V2.2 
            # have to compare performance 
            # self.network = SVR_AtlasNet() 

        elif self.opt.network == 'foldnet':
            self.network = FoldNet(template_type=self.opt.template_type, 
                                    num_points=self.opt.number_points,
                                    bottleneck_size=self.opt.bottleneck_size, 
                                    hidden_neurons=self.opt.hidden_neurons,
                                    pretrained_encoder=False,
                                    remove_all_batchNorms=self.opt.remove_all_batchNorms)
            
            self.network.to(self.opt.device)
        else:
            raise NotImplementedError(f"{self.network} is not implemented/imported")
        
        if self.opt.parallel:
            self.network = nn.DataParallel(self.network, device_ids=self.opt.multi_gpu)

        f = open(os.path.join(self.opt.dir_name, 'arch.txt'), 'w')
        f.write(repr(self.network))

        self.reload_network()

    def reload_network(self):
        """
        Reload entire model or only decoder (atlasnet) depending on the options
        :return:
        """
        if self.opt.reload_model_path != "":
            self.opt.logger.info(f"Network weights loaded from  {self.opt.reload_model_path}!")

            self.network.load_state_dict(torch.load(self.opt.reload_model_path, map_location='cuda'))

        elif self.opt.reload_decoder_path != "":
            opt = deepcopy(self.opt)
            opt.SVR = False
            network = EncoderDecoder(opt)
            network.module.load_state_dict(torch.load(opt.reload_decoder_path, map_location='cuda'))
            self.network.module.decoder = network.module.decoder
            self.opt.logger.info(f"Network Decoder weights loaded from  {self.opt.reload_decoder_path}!")

        else:
            self.opt.logger.info("No network weights to reload!")

    def build_optimizer(self):
        """
        Create optimizer
        """
        if self.opt.train_only_encoder:
            # To train a resnet image encoder with a pre-trained atlasnet decoder.
            self.opt.logger.info("only train the Encoder")
            self.optimizer = optim.Adam(self.network.module.encoder.parameters(), lr=self.opt.lrate)
        else:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)

        if self.opt.reload_optimizer_path != "":
            try:
                self.optimizer.load_state_dict(torch.load(self.opt.reload_optimizer_path, map_location='cuda'))
                # self.opt.logger.info(f"Reloaded optimizer {self.opt.reload_optimizer_path}")
            except:
                self.opt.logger.info(f"Failed to reload optimizer {self.opt.reload_optimizer_path}")

        # Set policy for warm-up if you use multiple GPUs
        # self.next_learning_rates = []
        # if len(self.opt.multi_gpu) > 1:
        #     self.next_learning_rates = np.linspace(self.opt.lrate, self.opt.lrate * len(self.opt.multi_gpu),
        #                                            5).tolist()
        #     self.next_learning_rates.reverse()
