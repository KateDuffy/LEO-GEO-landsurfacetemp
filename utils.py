import torch
import torch.nn.init as init

import numpy as np
import yaml

def get_sensor_stats(sensor):
    '''
    Values computed by scripts/get_data_stats.py
    inputs
        sensor: (AHI, ABI, G16, G17, H8)
    output
        mean
        standard deviation
    '''
    if sensor in ['AHI', 'H8', 'H8_15']:
        # computed from himawari
        mu = (3.237, 3.219, 3.227, 3.231, 3.199, 3.166, 284.887, 236.479, 245.53, 
              252.933, 274.039, 254.92, 276.217, 275.291, 272.772, 260.83)
        sd = (3.051, 3.068, 3.08, 3.058, 3.105, 3.135, 18.46, 8.544, 10.155, 11.508, 
              19.126, 15.423, 19.826, 20.088, 19.096, 15.088)
    elif sensor in ['ABI', 'G16', 'G17']:
        mu = (0.242, 0.191, 0.348, 0.0187, 0.268, 0.184, 299.916, 239.751, 249.354,
              258.246, 287.677, 263.352, 290.921, 290.172, 287.110, 270.647)
        sd = (0.254, 0.250, 0.245, 0.207, 0.231, 0.223, 13.349, 7.829, 8.599, 9.429, 
              14.894, 12.998, 14.880, 15.510, 14.460, 11.167)
    elif sensor in ["terra"]:
        mu = (300.132,)
        sd = (11.599,)
    elif sensor in ["aqua"]:
        mu = (300.132,) #(290.711,)
        sd = (11.599,) #(14.342,)
    elif sensor in ["MODIS_L1B"]:
        mu = (287.581, 297.169, 296.787, 286.871, 284.287,) #band 31, 21, 22, 32, 29
        sd = (12.889, 13.254, 13.333, 12.747, 12.023,)
        
    return mu, sd

def make_patches(x, patch_size):
    h, w, c = x.shape
    r = list(range(0, h, patch_size))
    r[-1] = h - patch_size
    samples = [x[np.newaxis,i:i+patch_size, j:j+patch_size] for i in r for j in r]
    samples = np.concatenate(samples, 0)
    return samples

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

def scale_image(x):
    xmn = torch.min(x)
    xmx = torch.max(x)
    return (x - xmn) / (xmx - xmn)

def get_config(yaml_file):
    '''
    Args:
        yaml_file (str): configuration file like configs/Base-G16G18.yaml
    Returns:
        (dict): dictionary of parameters
    '''
    with open(yaml_file) as f:
        return yaml.load(f)
    
    
def unnormalize(x, mu, sd):
    x = x * torch.Tensor(sd) + torch.Tensor(mu)
    return x

