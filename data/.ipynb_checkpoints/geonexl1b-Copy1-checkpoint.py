import os, sys
import numpy as np
from pyhdf.SD import SD, SDC
from scipy import ndimage
import glob
import pandas as pd
import xarray as xr
import rioxarray as rxr


'''
# Basic parameters
lat_0 = 60
lon_0 = -180
res_x = 0.01                # 0.02 for the 2km grid
res_y = 0.01                # 0.02 for the 2km grid
tile_xdim = 600            # 300 for the 2km grid
tile_ydim = 600            # 300 for the 2km grid

# Input information
hid = 14                       # 0 - 59
vid =  5                        # 0 - 19
x = 0                            # column/sample, 0-(tile_xdim-1) 
y = 0                            # row/line, 0-(tile_ydim-1) 

# Output formula 
lat_ulcnr = lat_0 - (vid*tile_ydim + y)*res_y        # upper-left corner latitude
lon_ulcnr = lon_0 + (hid*tile_xdim + x)*res_y     # upper-left corner longitude
lat_cntr = lat_ulcnr - 0.5*res_y                            # center latitude
lon_cntr = lon_ulcnr + 0.5*res_x                        # center longitude
'''

def get_tile_coords(tile, resolution_km=2.):
    hid = int(tile[1:3])
    vid = int(tile[4:6])
    
    lat_0 = 60
    lon_0 = -180
    res_x = 0.01 * resolution_km
    res_y = 0.01 * resolution_km
    tile_xdim = int(600 / resolution_km)
    tile_ydim = int(600 / resolution_km)
    
    lat_ulcnr = lat_0 - vid*tile_ydim*res_y     # upper-left corner latitude
    lon_ulcnr = lon_0 + hid*tile_xdim*res_y     # upper-left corner longitude
    lat_cntr = lat_ulcnr - 0.5*res_y            # center latitude
    lon_cntr = lon_ulcnr + 0.5*res_x    
    lats = np.linspace(lat_0 - vid*tile_ydim*res_y, lat_0 - (vid+1)*tile_ydim*res_y+res_y, tile_ydim, endpoint=True)
    lons = np.linspace(lon_0 + hid*tile_xdim*res_x, lon_0 + (hid+1)*tile_xdim*res_x-res_x, tile_xdim, endpoint=True)    
    return lats, lons

    
    
class L1BFile(object):
    '''
    Reads a single L1B file at a common resolution. Channels are bilinearly interpolated to the defined resolution.
    Args:
        file: Filepath to L1b
        bands (optional): List of bands, default=list(range(1,17))
        resolution_km (optional): Resolution in km for common grid, default=2
    '''
    def __init__(self, file, bands=list((range(1,17))), resolution_km=2.):
        self.file = file
        self.bands = bands
        self.band_names =  list(range(7,17))
        self.resolution_km = resolution_km
        self.reflective_bands = list(range(1,7))
        self.emissive_bands = list(range(7,17))

        
#     def __init__(self, file, bands=list(range(1, 26)), resolution_km=2.):
#         self.file = file
#         self.bands = bands
#         self.band_names =  list(range(5, 26))
#         self.resolution_km = resolution_km
#         self.resolution_size = int(600. / resolution_km)
    
    def load(self):
        f = rxr.open_rasterio(self.file, masked=False)
        data_array = np.zeros((self.resolution_size, self.resolution_size, len(self.bands)))
        for i, b in enumerate(self.bands):
            arr = f.griddedL1B.sel(band=b)
            if arr.shape[0] != 300:
                 arr = ndimage.interpolation.zoom(arr, self.resolution_size/arr.shape[0], order=1)
            arr[arr <= 0] = np.nan
            data_array[:,:,i] = arr
        return data_array
    
    
    
    def load_xarray(self):
        data = self.load()
        tile = self.file.split("/")[-3]
        lats, lons = get_tile_coords(tile)
        
        da = xr.DataArray(data[:,:,4:], dims=('lat', 'lon', 'band'), coords=dict(lat=lats, lon=lons, band=self.band_names))
        SZA = xr.DataArray(data[:,:,0], dims=('lat', 'lon'), coords=dict(lat=lats, lon=lons))
        VZA = xr.DataArray(data[:,:,1], dims=('lat', 'lon'), coords=dict(lat=lats, lon=lons))
        SAZ = xr.DataArray(data[:,:,2], dims=('lat', 'lon'), coords=dict(lat=lats, lon=lons))
        VAZ = xr.DataArray(data[:,:,3], dims=('lat', 'lon'), coords=dict(lat=lats, lon=lons))
        return xr.Dataset({'L1': da, 'abi_SZA': SZA, 'abi_VZA': VZA, 'abi_SAZ': SAZ, 'abi_VAZ': VAZ})



class ModisL1B(object):
    '''
    Get information on L1B data directory, available tiles, years, and files
        file lists are locally cached to future reading as retrieving file lists
        can be time consuming.
    Args:
        data_directory: directory of the L1B product
        sensor: (Terra, Aqua)
    '''
    def __init__(self, data_directory, sensor):
        self.data_directory = data_directory
        self.sensor = sensor
        self.sat = os.path.basename(os.path.dirname(os.path.dirname(data_directory)))

    def tiles(self):
        tile_pattern = os.path.join(self.data_directory, 'h*v*')
        tile_folders = glob.glob(tile_pattern)
        tiles = [os.path.basename(t) for t in tile_folders]
        return tiles

    def years(self):
        tile = self.tiles()[0]
        years = os.listdir(os.path.join(self.data_directory, tile))
        years = [int(y) for y in years if y[0] == '2']
        return years

    def hours(self):
        return list(range(0,24))

    def files(self, tile=None, year=None, dayofyear=None, cachedir='.tmp'):
        '''
        Args:
            tile (optional): Tile from GeoNEX grid
            year (optional): Year of files to get
            dayofyear (optional): Day of year
            cachedir (optional): Cache filelist in directory
        Returns:
            pd.DataFrame of filelist with year, dayofyear, hour, minute, tile, file, h, and v
        '''
        if tile == None:
            tile = '*'
        if year == None:
            year = '*'
        else:
            year = str(year)
        if dayofyear == None:
            dayofyear = '*'
        else:
            dayofyear = '%03i' % dayofyear

#         cache_file = f'{cachedir}/filelist/{self.sat}_{self.sensor}_{tile}_{year}_{dayofyear}.pkl'
#         if os.path.exists(cache_file):
#             return pd.read_pickle(cache_file)


        file_pattern = os.path.join(self.data_directory, '%s/%s/*.hdf' % (tile, year))
        files = glob.glob(file_pattern)
        fileinfo = []
        for f in files:
            fl = os.path.basename(f)
            y = fl.split(".")[2][:4]
            doy = fl.split(".")[2][4:7]
            hour = fl.split(".")[2][7:9]
            minute = fl.split(".")[2][9:11]
            tile = fl.split(".")[1]
            h, v = int(tile[0:2]), int(tile[2:4])


            fileinfo.append(dict(year=y, dayofyear=doy, hour=hour,
                              minute=minute, file=f, tile=tile, h=h, v=v))
        fileinfo = pd.DataFrame(fileinfo)
#         if not os.path.exists(os.path.dirname(cache_file)):
#             os.makedirs(os.path.dirname(cache_file))
#         fileinfo.to_pickle(cache_file)
        return fileinfo


