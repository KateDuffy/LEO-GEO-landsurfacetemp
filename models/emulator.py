# https://github.com/tjvandal/nex-ai-geo-translation

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import glob
import numpy as np

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass


from utils import get_sensor_stats, scale_image



class MAIACEmulatorCNN(nn.Module):
    def __init__(self, in_ch, out_ch, hidden=512, tau=1e-3, priorlengthscale=1e1, N=int(1e9)):
        super(MAIACEmulatorCNN, self).__init__()
        self.out_ch = out_ch
        self.h0 = nn.Conv2d(in_ch, hidden, 3, padding=1, stride=1)
        self.h1 = nn.Conv2d(hidden, hidden, 3, padding=1, stride=1)
        self.h2 = nn.Conv2d(hidden, hidden, 3, padding=1, stride=1)
        self.h3 = nn.Conv2d(hidden, out_ch+1, 3, padding=1, stride=1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, train=True):
        x1  = self.h0(x)
        x1_1 = self.activation(x1)

        x2 = self.h1(x1_1)
        x2_1 = self.activation(x2)

        x3 = self.h2(x2_1)
        x3_1 = self.activation(x3)

        x4 = self.h3(x3_1 + x1_1)
        
        prob = self.sigmoid(x4[:,:1])
        mu = x4[:,1:self.out_ch + 1]
        logvar = x4[:,self.out_ch + 1:]
        return mu, logvar, prob


        
        
        
        
class MAIACTrainer(nn.Module):
    def __init__(self, params, distribute=False, rank=0, gpu=0):
        super(MAIACTrainer, self).__init__()
        self.params = params
        self.distribute = distribute

        # set model
        self.model = MAIACEmulatorCNN(params['input_dim'], 1, params['hidden']).to(gpu)
        self.model = self.model.cuda()
        self.checkpoint_filepath = os.path.join(params['model_path'], 'checkpoint.pth.tar')

        if self.distribute:
            self.model = DDP(self.model.to(gpu), device_ids=[gpu])
        else:
            self.model = self.model.to(gpu)
            
        print(self.model)
        
        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=0)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.global_step = 0
        self.tfwriter_train = SummaryWriter(os.path.join(params['model_path'], 'train', 'tfsummary'))
        self.tfwriter_valid = SummaryWriter(os.path.join(params['model_path'], 'valid', 'tfsummary'))

    def load_checkpoint(self):
        filename = self.checkpoint_filepath
        if os.path.isfile(filename):
            print("loading checkpoint %s" % filename)
            checkpoint = torch.load(filename)
            self.global_step = checkpoint['global_step']
            if self.distribute:
                self.model.module.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (Step {})"
                    .format(filename, self.global_step))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    def save_checkpoint(self):
        if self.distribute:
            state = {'global_step': self.global_step, 
                     'model': self.model.module.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
        else:
            state = {'global_step': self.global_step, 
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
        torch.save(state, self.checkpoint_filepath)

    def step(self, x, y, mask, train=True, log=False):
        
        if self.distribute:
            model = self.model.module
        else:
            model = self.model
            
        mask = torch.ones_like(y) * mask
        
        y *= mask
        eps = 1e-7
        y_hat, logvar, y_prob = model(x, train=train)
        
        y_cond = torch.masked_select(y, mask.bool())
        y_hat_cond = torch.masked_select(y_hat, mask.bool())
        #y_logvar_cond = torch.masked_select(logvar, mask.bool())
        #y_precision_cond = torch.exp(-y_logvar_cond) + eps
        
        #cond_logprob = y_precision_cond * (y_cond -  y_hat_cond) ** 2 + y_logvar_cond
        cond_logprob = (y_cond -  y_hat_cond) ** 2
        #cond_logprob = torch.abs(y_cond -  y_hat_cond)
        cond_logprob *= -1
        cond_logprob = torch.mean(cond_logprob)

        logprob_classifier = mask * torch.log(y_prob + eps) + (1-mask) * torch.log(1-y_prob + eps)
#         logprob_classifier = torch.masked_select(logprob_classifier, eval_mask.bool()) 
        logprob_classifier = torch.mean(logprob_classifier)
        logprob = logprob_classifier + cond_logprob

        neglogloss = -logprob

        loss = neglogloss

                
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if log:
            step = self.global_step
            if train:
                tfwriter = self.tfwriter_train
            else:
                tfwriter = self.tfwriter_valid
                
            tfwriter.add_scalar("losses/binary", -logprob_classifier, step)
            tfwriter.add_scalar("losses/regression", -cond_logprob, step)
            tfwriter.add_scalar("losses/loss", loss, step)

            y_hat *= (y_prob > 0.5).float()
            
            # create grid of images
            x_grid = torchvision.utils.make_grid(x[:4,[0,1]])
            y[y != y] = 0
            y_grid = torchvision.utils.make_grid(y[:4,[0]])
            mask_grid = torchvision.utils.make_grid(mask[:8,:1])

            seg_grid = torchvision.utils.make_grid(y_prob[:4,:1])
            y_reg_grid = torchvision.utils.make_grid(y_hat[:4,[0]])
            
            # accuracy
            predicted = (y_prob > 0.5).float()
            
            accuracy = torch.mean((predicted == mask).float())
            tfwriter.add_scalar("accuracy", accuracy, step)
            
            # write to tensorboard
            tfwriter.add_image('inputs', scale_image(x_grid), step)
            tfwriter.add_image('label', scale_image(y_grid), step)
            tfwriter.add_image('regression', scale_image(y_reg_grid), step)

            tfwriter.add_image('mask', mask_grid, step)
            tfwriter.add_image('segmentation', seg_grid, step)
            try:
                tfwriter.add_histogram('segmentation', mask, step)
                tfwriter.add_histogram('cond_observed', y_cond, step)
                tfwriter.add_histogram('cond_regression', y_hat_cond, step)
            except:
                pass
            
        if train:
            self.global_step += 1

        return loss

