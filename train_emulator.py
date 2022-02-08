# https://github.com/tjvandal/nex-ai-geo-translation
'''
Training script to emulate modis mod11a1 using a CNN and L1G bands
'''
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils import data

import argparse
from torchvision import transforms
import torch
from models import emulator
import time, os
import numpy as np
import random

from places import Places
import utils



class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples
    def __iter__(self):
        return iter(self.loop())
    def __len__(self):
        return 2 ** 31
    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples) 
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


def transform_sensor(sensor):
    mu, sd = utils.get_sensor_stats(sensor)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mu, std=sd)])
    return tf

def rotate_flip(x,y,mask):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rotate = random.random() < 0.5
    if hflip:
        x = torch.flip(x, [2,3])
        y = torch.flip(y, [2,3])
        mask = torch.flip(mask, [2,3])
    if vflip:
        x = torch.flip(x, [3,2])
        y = torch.flip(y, [3,2])
        mask = torch.flip(mask, [3,2])
    if rotate:
        x = torch.rot90(x, 1, [2,3])
        y = torch.rot90(y, 1, [2,3]) 
        mask = torch.rot90(mask, 1, [2,3]) 
        
    return x, y, mask


    
def train_net(params, rank=None, device=None, distribute=False):
    if rank is None:
        rank = 0
#         setup(rank, 1, 9100+rank)
        
    if device is None:
        device = 0 #rank % torch.cuda.device_count()
        
    trainer = emulator.MAIACTrainer(params, distribute=distribute)
    
    # set device
    trainer.to(device)
    if rank == 0:
        trainer.load_checkpoint()
    
    tf_ABI, tf_MOD = transform_sensor("ABI"), transform_sensor("terra")
    tf_mask = transforms.Compose([transforms.ToTensor()])
    
    dataset_train = Places(params['file_path_train'], tf_ABI, tf_MOD, tf_mask, params['bands'])
    train_iter = iter(data.DataLoader(dataset_train, batch_size=params['batch_size'], num_workers=8,
                                      sampler=InfiniteSampler(len(dataset_train))))
    dataset_valid = Places(params['file_path_valid'], tf_ABI, tf_MOD, tf_mask, params['bands'])
    valid_iter = iter(data.DataLoader(dataset_valid, batch_size=params['batch_size'], num_workers=8,
                                      sampler=InfiniteSampler(len(dataset_valid))))
    
    while 1:
        x, y, mask = [x.type(torch.cuda.FloatTensor).to(device) for x in next(train_iter)]
        x, y, mask = rotate_flip(x, y, mask)
        
        log = False
        if (trainer.global_step % params['log_iter'] == 0) and (rank == 0):
            log = True
            

        train_loss = trainer.step(x, y, mask, log=log, train=True)
        if log:
            x, y, mask = [x.type(torch.cuda.FloatTensor).to(device) for x in next(valid_iter)]
            x, y, mask = rotate_flip(x, y, mask)

            valid_loss = trainer.step(x, y, mask, log=log, train=False)
            print(f"Step: {trainer.global_step}\tTrain Loss: {train_loss.item():4.4g}\tValid Loss: {valid_loss.item():4.4g}")

        if (rank == 0) and (trainer.global_step % params['checkpoint_step'] == 1):
            trainer.save_checkpoint()

        if trainer.global_step >= params['max_iter']:
            break

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{port}'
    
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def train_net_mp(rank, world_size, port, params):
    setup(rank, world_size, port)
    train_net(params, rank=rank, distribute=True)
    
def run_training(params, world_size, port):
    params['batch_size'] = params['batch_size'] // world_size
    mp.spawn(train_net_mp,
             args=(world_size, port, params,),
             nprocs=world_size,
             join=True)
    cleanup()
    

if __name__ == '__main__':
    params = {'file_path_train': '/nobackupp13/kmduffy1/cross_sensor_training/patches_terra_G16/2019/',
              'file_path_valid': '/nobackupp13/kmduffy1/cross_sensor_training/patches_terra_G16/2020/',
              'model_path': '/nobackupp13/kmduffy1/cross_sensor_training/models/mod11a1/L1G_terra_b7to16_128h_2019/',
              'lr': 1e-4,
              'max_iter': int(3e5),
              'checkpoint_step': 1000,
              'log_iter': 500,
              'bands': [7,8,9,10,11,12,13,14,15,16],
              'input_dim': 11,
              'world_size': 4,
              'port': 9001,
              'batch_size': 32,
              'hidden': 128}
    
    train_net(params)
