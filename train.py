import os
import sys
import auxiliary.argument_parser as argument_parser
import auxiliary.my_utils as my_utils
import time
import torch
from auxiliary.my_utils import yellow_print
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


opt = argument_parser.parser()
my_utils.plant_seeds(manual_seed=opt.manual_seed)
import training.trainer as trainer
trainer = trainer.Trainer(opt)
trainer.build_dataset()
trainer.build_network()
trainer.build_optimizer()
trainer.build_losses()
trainer.start_train_time = time.time()

if opt.demo:
    with torch.no_grad():
        trainer.demo(opt.demo_input_path)
    sys.exit(0)

if opt.run_single_eval:
    with torch.no_grad():
        trainer.test_epoch()
    sys.exit(0)

for epoch in range(trainer.epoch, opt.nepoch):
    trainer.train_epoch()
    with torch.no_grad():
        trainer.test_epoch()
    trainer.dump_stats()
    trainer.increment_epoch()
    trainer.save_network()


opt.logger.info(f"Training time {(time.time() - trainer.start_time)//60} minutes.")

# f = open(os.path.join(opt.dir_name, opt.train_endsignal), "w")
# f.write("Now the training end!")
# f.close()



