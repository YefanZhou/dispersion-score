#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs. 
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.


import threading
import time
import os
import sys
import numpy as np
import gpustat
import logging
import itertools

exitFlag = 0
GPU_MEMORY_THRESHOLD = 500 # MB?
AVAILABLE_GPUS = [0, 1, 2, 3]   #[0, 1, 2, 3, 4, 5, 6, 7]
MAX_NCHECK=10              # number of checks to know if gpu free

## If we need to wait for the entire clean cluster to start, select False here

all_empty = {"ind": True}
#all_empty = {"ind": False}

def num_available_GPUs(gpus):
    
    sum_i = 0
    for i, stat in enumerate(gpus):
        if stat['memory.used'] < 100:
            sum_i += 1
    return sum_i


def get_free_gpu_indices(logger):
    '''
        Return an available GPU index.
    '''
    counter = {}
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        # print('stats length: ', len(stats))
        
        if num_available_GPUs(stats.gpus) >= 4:
            all_empty["ind"] = True
            
        if not all_empty["ind"]:
            logger.info("Previous experiments not finished...")
            time.sleep(10)
            continue
        
        max_checks = 0
        max_gpu_id = -1
        for i, stat in enumerate(stats.gpus):
            memory_used = stat['memory.used']
            if memory_used < GPU_MEMORY_THRESHOLD and i in AVAILABLE_GPUS:
                if i not in counter:
                    counter.update({i: 0})
                else:
                    counter[i] = counter[i] + 1
                ###Multiple Check available to avoid some accident 
                if counter[i] >= MAX_NCHECK:
                    return i
            else:
                counter.update({i: 0})

            if counter[i] > max_checks:
                max_checks = counter[i]
                max_gpu_id = i

        print(f"Waiting on GPUs, Checking {max_checks}/{MAX_NCHECK} at gpu {max_gpu_id}")
        time.sleep(10)

        
class DispatchThread(threading.Thread):
    def __init__(self, name, bash_command_list, logger, gpu_m_th, gpu_list, maxcheck):
        threading.Thread.__init__(self)
        self.name = name
        self.bash_command_list = bash_command_list
        self.logger = logger
        global GPU_MEMORY_THRESHOLD
        GPU_MEMORY_THRESHOLD = gpu_m_th
        global AVAILABLE_GPUS
        AVAILABLE_GPUS = gpu_list
        global MAX_NCHECK
        MAX_NCHECK = maxcheck

        

    def run(self):
        self.logger.info("Starting " + self.name)
        threads = []
        for i, bash_command in enumerate(self.bash_command_list):
             
            import time
                
            time.sleep(0.3)
            
            #if os.path.isfile(result_name):
            #    print("Result already exists! {0}".format(result_name))
            #    continue
            #    
            #else:
            #    print("Result not ready yet. Running it for a second time: {0}".format(result_name))
            
            cuda_device = get_free_gpu_indices(self.logger)
            thread1 = ChildThread(f"{i}th + {bash_command}", 1, cuda_device, bash_command, self.logger)
            thread1.start()
            
            time.sleep(30)
            threads.append(thread1)

        # join all.
        for t in threads:
            t.join()
        self.logger.info("Exiting " + self.name)


class ChildThread(threading.Thread):
    def __init__(self, name, counter, cuda_device, bash_command, logger):
        threading.Thread.__init__(self)
        self.name = name
        self.counter = counter
        self.cuda_device = cuda_device
        self.bash_command = bash_command
        self.logger = logger

    def run(self):
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device[0]},{self.cuda_device[1]}'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device}'
        bash_command = self.bash_command

        self.logger.info(f'executing {bash_command} on GPU: {self.cuda_device}')
        # ACTIVATE
        os.system(bash_command)
        import time
        import random
        time.sleep(random.random() % 5)

        self.logger.info("Finishing " + self.name)      


def get_logger(path, fname):
    if not os.path.exists(path):
        os.mkdir(path)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_log_handler = logging.FileHandler(os.path.join(path, fname))
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(file_log_handler)
    logger.addHandler(stderr_log_handler)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s","%Y-%m-%d %H:%M:%S")
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)
    sys.stdout.flush()

    return logger

