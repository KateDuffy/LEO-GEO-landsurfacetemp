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
    Args:
        config_file: get parameters and model_directory from configuration file
        device: set device for inference
    Return:
        MAIACEmulatorCNN (torch.nn.Module)
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
    Split a 3D numpy array into patches for inference
    (Channels, Height, Width)
    Args:
        tile_size: width and height of patches to return
        overlap: number of pixels to overlap between patches
    Returns:
        dict(patches, upper_left): patches and indices of original array
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
    
#     x.requires_grad_()
    y_hat, sigma, y_prob = model(torch.unsqueeze(x, 0).type(torch.FloatTensor).to("cuda:0"), train=False)
    
    # Do backpropagation to get the derivative of the output based on the image
#     saliency = y_hat.backward()
    
    out = {'loc':np.squeeze(y_hat, axis=-1),
           "probs": np.squeeze(y_prob, axis=-1)}
    return out



def single_inference_split(X, model, sensor, patch_size=64, overlap=10, discard=0):
    
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
    
        y1, y2 = np.min(ds.lat.values)-1, np.max(ds.lat.values)+1
        x1, x2 = np.min(ds.lon.values)-1, np.max(ds.lon.values)+1
        elevation_patch =  elevation.sel(y=slice(y1, y2)).sel(x=slice(x1, x2))

        elevation_patch.load()
        elevation_patch = elevation_patch.interpolate_na(dim="x", method="linear")
        elevation_patch = elevation_patch.interp(y=ds.lat, x=ds.lon)
        return elevation_patch.z.values/8518.0



def inference_validation_pairs(params, LEO_sensor, GEO_sensor, files):

    model = params["model_path"].split("/")[-2]
    save_dir = "/nobackupp13/kmduffy1/cross_sensor_training/inference/%s/%s/" %(model, params["save_folder"]) 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = load_model(params)
    tf_ABI, tf_MOD = transform_sensor("G16"), transform_sensor(LEO_sensor)
    tf_mask = transforms.Compose([transforms.ToTensor()])

    for tile in tqdm.tqdm(files):
        
        if not os.path.exists(save_dir + tile.split("/")[-1]):
        
            ds = xr.open_mfdataset(tile)
            #if (np.sum(ds.L1.values != ds.L1.values) == 0) & (np.nansum(ds.L1.sel(band=slice(7,16)).values < 1e-6) < 1): # check for missing L1 data
            if (np.nansum(ds.L1.sel(band=slice(7,16)).values < 1e-6) < 1):
                geo_bands = ds.L1.values
                geo_bands = tf_ABI(geo_bands)
                bands = [b-1 for b in params['bands']]
                geo_bands = geo_bands[bands,:,:] 

                elevation = ds.elevation.values
                elevation = tf_mask(elevation)
                geo_bands = torch.cat((geo_bands, elevation), dim=0)


                try:
                    out = single_inference_split(geo_bands, model, LEO_sensor)

                    ds["loc"] = (("lat", "lon"), out["loc"][0,:,:])
                    ds["probs"] = (("lat", "lon"), out["probs"][0,:,:])

                    ds.to_netcdf(save_dir + tile.split("/")[-1])
                    print("saved ", save_dir + tile.split("/")[-1])
                except:
                    print("passed ", save_dir + tile.split("/")[-1])
            else:
                print("bad L1 data ", tile.split("/")[-1])

                

                
def inference_G16(params, tile="h11v04", year=2020, doy=1, sensor="G16"):  
    
    save_directory = "/nobackupp13/kmduffy1/NEXAI-LST/%s/%04d/%03d/" % (tile, year, doy)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    directory = '/nex/datapool/geonex/public/GOES16/GEONEX-L1G/'
    geo = geonexl1g.GeoNEXL1G(directory, sensor)
    files = geo.files(tile=tile, year=year, dayofyear=doy)
    
    
    if len(files) == len(glob.glob(save_directory+"*")):
        print("Done")
    else:
    
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


            
if __name__ == "__main__":

    LEO_sensor = "aqua"
    GEO_sensor = "G16"

    data_dir = "/nobackupp13/kmduffy1/cross_sensor_training/tiles_%s_%s/2020/*" %(LEO_sensor, GEO_sensor)
    files = glob.glob(data_dir)
    shuffle(files)
    print(len(files))


    params = {'model_path':'/nobackupp13/kmduffy1/cross_sensor_training/models/mod11a1/L1G_terra_b7to16_128h_2019/',
              'save_folder': data_dir.split("/")[-3].split("tiles_")[-1],
              'bands': [7,8,9,10,11,12,13,14,15,16],
              'input_dim': 11,
              'hidden': 128,
              'batch_size':1}

    inference_validation_pairs(params, LEO_sensor, GEO_sensor, files[:10])
    
    



    
