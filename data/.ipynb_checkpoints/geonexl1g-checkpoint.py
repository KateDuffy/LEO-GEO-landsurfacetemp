import os, sys
import numpy as np
from pyhdf.SD import SD, SDC
from scipy import ndimage
import glob
import pandas as pd
import xarray as xr


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
    
    lat_ulcnr = lat_0 - vid*tile_ydim*res_y        # upper-left corner latitude
    lon_ulcnr = lon_0 + hid*tile_xdim*res_y     # upper-left corner longitude
    lat_cntr = lat_ulcnr - 0.5*res_y                            # center latitude
    lon_cntr = lon_ulcnr + 0.5*res_x    
    lats = np.linspace(lat_0 - vid*tile_ydim*res_y, lat_0 - (vid+1)*tile_ydim*res_y+res_y, tile_ydim, endpoint=True)
    lons = np.linspace(lon_0 + hid*tile_xdim*res_x, lon_0 + (hid+1)*tile_xdim*res_x-res_x, tile_xdim, endpoint=True)    
    return lats, lons

class L1GFile(object):
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
        self.resolution_km = resolution_km
        self.resolution_size = int(600. / resolution_km)
        self.reflective_bands = list(range(1,7))
        self.emissive_bands = list(range(7,17))
        

    def load(self):
        fp = SD(self.file, SDC.READ)
        data_array = np.zeros((self.resolution_size, self.resolution_size, len(self.bands)))
        for i, b in enumerate(self.bands):
            b_obj = fp.select('BAND%02i' % b)
            attrs = b_obj.attributes()
            scale_factor = attrs['Scale_Factor']
            #fill_value = attrs['_FillValue'] ## this fill value seems to be wrong in l1g
            fill_value = 32768.
            arr = b_obj.get()[:].astype(np.float32)
            offset = attrs['Offset_Constant']
            arr[arr == fill_value] = np.nan
            arr *= scale_factor
            arr += offset
            if arr.shape[0] != self.resolution_size:
                arr = ndimage.interpolation.zoom(arr, self.resolution_size/arr.shape[0], order=1)
            data_array[:,:,i] = arr
        #self.data_array = data_array
        return data_array

    def solar(self):
        fp = SD(self.file, SDC.READ)
        sa = fp.select('Solar_Azimuth').get()[:]
        sz = fp.select('Solar_Zenith').get()[:]
        if sa.shape[0] != self.resolution_size:
            sa = ndimage.interpolation.zoom(sa, self.resolution_size/sa.shape[0], order=1)
            sz = ndimage.interpolation.zoom(sz, self.resolution_size/sz.shape[0], order=1)
        return sa*0.01, sz*0.01
    
    def load_xarray(self):
        data = self.load()
        fp = SD(self.file, SDC.READ)
        try:
            tile_path = os.path.join("/".join(self.file.split("/")[:-3]), 'constant')
            tile_file = glob.glob(tile_path + '/*.hdf')[0]
        except:
            tile_path = os.path.join('/nex/datapool/geonex/public/GOES16/GEONEX-L1G/', self.file.split("/")[7], 'constant')
            tile_file = glob.glob(tile_path + '/*.hdf')[0]
            
        sat, sensor, date, time, _, tile, _ = os.path.basename(self.file).replace(".hdf","").split("_")
        lats, lons = get_tile_coords(tile)
        sa, sz = self.solar()
        # print(tile, 'lats', len(lats), 'data', data.shape)
        da = xr.DataArray(data, dims=('lat', 'lon', 'band'), coords=dict(lat=lats, lon=lons, band=self.bands))
        zenith = xr.DataArray(sz, dims=('lat', 'lon'), coords=dict(lat=lats, lon=lons))
        azimuth = xr.DataArray(sa, dims=('lat', 'lon'), coords=dict(lat=lats, lon=lons))
        return xr.Dataset({'L1': da, 'zenith': zenith, 'azimuth': azimuth})
        
class GeoNEXL1G(object):
    '''
    Get information on L1G data directory, available tiles, years, and files
        file lists are locally cached to future reading as retrieving file lists
        can be time consuming.
    Args:
        data_directory: directory of the L1G product
        sensor: (G16,G17,H8)
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

        
        file_pattern = os.path.join(self.data_directory, '%s/%s/%s/*.hdf' % (tile, year, dayofyear))
        files = glob.glob(file_pattern)
        fileinfo = []
        for f in files:
            root = os.path.dirname(f)
            rl = root.split('/')
            doy = int(rl[-1])
            y = int(rl[-2])

            fl = os.path.basename(f).split('_')
            hour = int(fl[3][:2])
            minute = int(fl[3][2:])
            tile = fl[5]
            h = int(tile[1:3])
            v = int(tile[4:6])
            
            fileinfo.append(dict(year=y, dayofyear=doy, hour=hour,
                              minute=minute, file=f, tile=tile, h=h, v=v))
        fileinfo = pd.DataFrame(fileinfo)
#         if not os.path.exists(os.path.dirname(cache_file)):
#             os.makedirs(os.path.dirname(cache_file))
#         fileinfo.to_pickle(cache_file)
        return fileinfo




class L1GPaired(object):
    def __init__(self, data_path1, data_path2, sensor1, sensor2):
        '''
        Retrieve lists of files that match temporally and spatially using two GeoNEXL1b objects
        
        Args:
            data_path1: First L1G data directory 
            data_path2: Second L1G data directory 
            sensor1: First sensor
            sensor2: Second sensor
        '''
        self.data_path1 = data_path1
        self.data_path2 = data_path2
        
        self.sensor1 = sensor1
        self.sensor2 = sensor2

        self.data1 = GeoNEXL1G(data_path1, sensor1)
        self.data2 = GeoNEXL1G(data_path2, sensor2)

    def tiles(self):
        tiles1 = self.data1.tiles()
        tiles2 = self.data2.tiles()
        intersect = np.intersect1d(tiles1, tiles2)
        return intersect.tolist()

    def day_pairs(self, year):
        tile = self.tiles()[0]
        path1 = os.path.join(self.data_path1, tile, str(year))
        return [int(day) for day in os.listdir(path1)]

    def files(self, tile=None, year=None, dayofyear=None, how='inner', cachedir='.tmp'):
        '''
        Get filelists for each data directory and join in space-time 
        
        Args:
            tile (optional): Tile from GeoNEX grid
            year (optional): Year of files to get
            dayofyear (optional): Day of year
            how (optional): Pandas method to join DataFrames, default='inner'
            cachedir (optional): Cache filelist in directory
        Returns:
            pd.DataFrame of filelist with year, dayofyear, hour, minute, tile, file, h, and v
        '''
        files1 = self.data1.files(tile=tile, year=year, dayofyear=dayofyear, cachedir=cachedir)
        files2 = self.data2.files(tile=tile, year=year, dayofyear=dayofyear, cachedir=cachedir)
        get_timestamp = lambda x: '_'.join(os.path.basename(x).split('_')[2:4])

        files1['timestamp'] = files1['file'].apply(get_timestamp)
        files1 = files1.set_index(['timestamp', 'tile'])

        files2['timestamp'] = files2['file'].apply(get_timestamp)
        files2 = files2.set_index(['timestamp', 'tile'])
        joined = files1.join(files2, lsuffix='1', rsuffix='2', how=how)
        return joined
    

class L1GMOSAIC(object):
    def __init__(self, data_path, sensor):
        self.data_path = data_path
        self.sensor = sensor
