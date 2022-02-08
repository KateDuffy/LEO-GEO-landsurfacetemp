import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import torch
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import xarray as xr

import utils



class Places(torch.utils.data.Dataset):
    def __init__(self, img_root, tf_ABI, tf_MOD, tf_mask, bands=[14, 15]):
        super(Places, self).__init__()
        self.tf_ABI = tf_ABI
        self.tf_MOD = tf_MOD
        self.tf_mask = tf_mask
        self.bands = bands
        self.paths = glob.glob(img_root + "*")

    def __getitem__(self, index):
        
        ds = xr.open_dataset(self.paths[index])
        geo_bands = ds.L1.values
        LST = ds.mod_LST.values
        QC = ds.mod_QC.values
        elevation = ds.elevation.values
        ds.close() 
        ds = None
        
        # 0, 1 = land/clear sky, 10 = cloud, 11 = other not produced (water)
        mask = (QC <= 1) * 1.
    
        # to tensor
        geo_bands = self.tf_ABI(geo_bands)
        LST = self.tf_MOD(LST)
        elevation = self.tf_mask(elevation)
        mask = self.tf_mask(mask)   
  
        mask[LST != LST] = 0.
        LST[mask < 0.5] = 0.
        bands = [b-1 for b in self.bands]
        geo_bands = geo_bands[bands,:,:]            

        geo_bands = np.concatenate((geo_bands, elevation), axis=0)
        
        return geo_bands, LST, mask
    

    def __len__(self):
        return len(self.paths)


