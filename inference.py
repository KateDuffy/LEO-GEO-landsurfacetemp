import glob
import tqdm
import numpy as np
from functools import partial
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from random import shuffle
import torch
from torchvision import transforms
import datetime

from models import emulator
# from data.prepare_training_data_ABI import get_geonex_tile_from_latlon_l1g, read_MOD11A1, crop_and_interp
from train_emulator import transform_sensor
from utils import get_sensor_stats, unnormalize
from data import geonexl1g


def load_model(params, device=None):

    '''
    Load MAIACEmulatorCNN model for inference
    
    Parameters
    ----------
    params: dict
        Information needed to load model
    device: str
        Device to use for inferenece, such as "cpu" or "cuda:0"
    
    Returns
    ----------
    output: torch.nn.Module
        MAIACEmulatorCNN
        
    '''    
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = emulator.MAIACEmulatorCNN(params['input_dim'], 1, params['hidden'])    
    model.to(device)
    checkpoint_path = os.path.join(params['model_path'], 'checkpoint.pth.tar')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    step = checkpoint['global_step']
    print(f"Loaded model from step: {step}")
    return model



def split_array(arr, tile_size=64, overlap=16):
    '''
    Split a 3D numpy array into patches of shape (Channels, Height, Width) for inference 
    
    Parameters
    ----------
    arr: numpy.ndarray
        Data array to split into patches
    tile_size: int
        Width and height of patches to return
    overlap: int
        Number of pixels to overlap between patches
    

    Returns:
    ----------
    output: dict
        Dictonary, dict(patches, upper_left), of patches and indices of original array 
        
    '''
    
    arr = arr[np.newaxis]
    width, height = arr.shape[2:4]
    arrs = dict(patches=[], upper_left=[])
    for i in range(0, width, tile_size - overlap):
        for j in range(0, height, tile_size - overlap):
            i = min(i, width - tile_size)
            j = min(j, height - tile_size)
            arrs['patches'].append(arr[:,:, i:i+tile_size,j:j+tile_size])
            arrs['upper_left'].append([[i,j]])
            
    
    arrs['patches'] = torch.cat(arrs['patches'], 0)
    arrs['upper_left'] = np.concatenate(arrs['upper_left'])
    return arrs['patches'], arrs['upper_left']



def single_inference(x, model):
    '''
    Perform inference on a single patch
    
    Parameters
    ----------
    arr: torch.tensor
        Patch to perform inference
    model: torch.nn.Module
        MAIACEmulatorCNN

    Returns:
    ----------
    output: dict
        Dictonary of predicted LST ("loc") and clear sky probability ("probs")
        
    '''    

    y_hat, sigma, y_prob = model(torch.unsqueeze(x, 0).type(torch.FloatTensor).to("cuda:0"), train=False)
    
    out = {'loc':np.squeeze(y_hat, axis=-1),
           "probs": np.squeeze(y_prob, axis=-1)}
    return out



def single_inference_split(X, model, sensor, patch_size=64, overlap=10, discard=0):
    '''
    Perform inference on a larger image by splitting into patches
    
    Parameters
    ----------
    X: torch.tensor
        Image on which to perform inference
    model: torch.nn.Module
        MAIACEmulatorCNN
    sensor: str
        Name of geostationary sensor. Defaults to GOES-16
    patch_size: int
        Width and height of patches
    overlap: int
        Number of pixels to overlap between patches
    discard: int
        Number of pixels to discard at borders of image

    Returns:
    ----------
    output: dict
        Dictonary of predicted LST ("loc") and clear sky probability ("probs")
        
    '''  
    
    mu, sd = get_sensor_stats(sensor)
    X_split, upper_left_idxs = split_array(X, patch_size, overlap=overlap)

    
    # perform inference on patches
    height, width = X.shape[1:3]
    counter = np.zeros((1,height-discard*2, width-discard*2))
    res_sum = {}
    for i, x in enumerate(X_split):
        ix, iy = upper_left_idxs[i]
        res_i = single_inference(x, model)
                
        keys = res_i.keys()
        if i == 0:
            res_sum = {k: np.zeros((res_i[k].shape[0], height-discard*2, width-discard*2)) for k in keys}

        for var in keys:
            if discard > 0:
                if var == "loc":
                    res_i[var] = unnormalize(res_i[var][0,:, discard:-discard,discard:-discard].cpu(), mu, sd).detach().numpy()                    
                else:
                    res_i[var] = res_i[var][0,:, discard:-discard,discard:-discard].cpu().detach().numpy()
            else:
                if var == "loc":
                    res_i[var] = unnormalize(res_i[var][0,:,:,:].cpu(), mu, sd).detach().numpy()                    
                else:
                    res_i[var] = res_i[var][0,:,:,:].cpu().detach().numpy()
            
            res_sum[var][:,ix:ix+patch_size-discard*2,iy:iy+patch_size-discard*2:] += res_i[var]
        counter[:,ix:ix+patch_size-discard*2,iy:iy+patch_size-discard*2:] += 1.

    out = {}
    for var in res_sum.keys():
        out[var] = res_sum[var]/ counter
    return out




def get_elevation(ds, elevation):
    
    '''
    Project elevation information to same lat/lon grid as a given dataset
    
    Parameters
    ----------
    ds: xarray.Dataset
        Dataset with 1D lat and lon dimensions
    elevation: xarray.Dataset
        Elevation dataset with 

    Returns:
    ----------
    output: xarray.Dataset
        Dictonary of predicted LST ("loc") and clear sky probability ("probs")
        
    '''  
    
        y1, y2 = np.min(ds.lat.values)-1, np.max(ds.lat.values)+1
        x1, x2 = np.min(ds.lon.values)-1, np.max(ds.lon.values)+1
        elevation_patch =  elevation.sel(y=slice(y1, y2)).sel(x=slice(x1, x2))

        elevation_patch.load()
        elevation_patch = elevation_patch.interpolate_na(dim="x", method="linear")
        elevation_patch = elevation_patch.interp(y=ds.lat, x=ds.lon)
        return elevation_patch.z.values/8518.0

                
def inference_GEO(model_path, tile, year=2020, doy=1, sensor="G16"):  
    '''
    Peform LST inference on L1G data and save prediction
    
    Parameters
    ----------
    model_path: str
        Directory location of saved Pytorch model, checkpoint.pth.tar
    tile: string
        GeoNEX tile for which to perform inference, such as 'h08v01'
    year: int
        Year of observation to perform inference. Defaults to 2020.
    doy: int
        Day of year perform inference. Defaults to 1.
    sensor: str
        Name of geostationary sensor. Defaults to GOES-16
        
    '''    

    save_directory = "/nobackupp13/kmduffy1/NEXAI-LST/%s/%04d/%03d/" % (tile, year, doy)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    L1G_directory = '/nex/datapool/geonex/public/GOES16/GEONEX-L1G/'
    geo = geonexl1g.GeoNEXL1G(L1G_directory, sensor)
    files = geo.files(tile=tile, year=year, dayofyear=doy)
    
    
    if len(files) == len(glob.glob(save_directory+"*")):
        print("Done")
    else:
        params = {'model_path':model_path,
              'bands': [7,8,9,10,11,12,13,14,15,16],
              'input_dim': 11,
               'hidden': 128,
               'batch_size':1}
        
        model = load_model(params)
        tf_ABI, tf_mask = transform_sensor("G16"), transforms.Compose([transforms.ToTensor()])
        elevation = glob.glob("/nobackupp13/kmduffy1/SRTM30/*")
        elevation_ds = xr.open_mfdataset(elevation, combine="by_coords")



        for i in tqdm.tqdm(range(len(files))):
            file, year, doy, h, m = files['file'].values[i], files.year[i], files.dayofyear[i], files.hour[i], files.minute[i]
            geo_data = geonexl1g.L1GFile(files['file'].values[i], resolution_km=2.).load_xarray()
            timestamp = datetime.datetime(year, 1, 1, h, m) + datetime.timedelta(int(doy-1))

            if not os.path.exists(save_directory + os.path.splitext(os.path.basename(file))[0] + ".nc"): 

                if (np.nansum(geo_data.L1.sel(band=slice(7,16)).values < 1e-6) < 1):
                    try:
                        geo_bands = geo_data.L1.values
                        geo_bands = tf_ABI(geo_bands)
                        bands = [b-1 for b in params['bands']]
                        geo_bands = geo_bands[bands,:,:] 

                        elevation = get_elevation(geo_data, elevation_ds)
                        elevation = tf_mask(elevation)
                        geo_bands = torch.cat((geo_bands, elevation), dim=0)
                        
                        out = single_inference_split(geo_bands, model, "terra")
                        ds = geo_data.copy()
                        ds = ds.expand_dims(time=[timestamp])
                        lst, clear = out["loc"], out["probs"]
                        lst[clear<0.5] = np.nan
                        lst[elevation.detach().numpy()<0.] = np.nan
                        ds["LST_Kelvin"] = (("time", "lat", "lon"), lst)
                        ds["clear_sky_probability"] = (("time", "lat", "lon"), clear)

                        ds = ds.drop(["azimuth", "zenith", "L1", "band"])
                        ds.to_netcdf(save_directory + os.path.basename(file).replace("ABI05", "NEXAI-LST").replace("hdf", "nc"))
                    except:
                        print("error on ", file)
                else:
                    print("bad L1G on ", file)

