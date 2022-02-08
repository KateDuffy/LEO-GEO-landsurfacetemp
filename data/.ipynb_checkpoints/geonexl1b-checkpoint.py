import os, sys
import numpy as np
from pyhdf.SD import SD, SDC
from scipy import ndimage
import glob
import pandas as pd
import xarray as xr



def GOES_lat_lon(g16nc, lat_rad_1d, lon_rad_1d):
    # GOES-R projection info and retrieving relevant constants
    proj_info = g16nc.variables['goes_imager_projection']
    lon_origin = -75
    H = 35786023.0+6378137.0
    r_eq = 6378137.0
    r_pol = 6356752.31414

    # create meshgrid filled with radian angles
    lat_rad,lon_rad = np.meshgrid(lat_rad_1d,lon_rad_1d)

    # lat/lon calc routine from satellite radian angle vectors

    lambda_0 = (lon_origin*np.pi)/180.0

    a_var = np.power(np.sin(lat_rad),2.0) + (np.power(np.cos(lat_rad),2.0)*(np.power(np.cos(lon_rad),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(lon_rad),2.0))))
    b_var = -2.0*H*np.cos(lat_rad)*np.cos(lon_rad)
    c_var = (H**2.0)-(r_eq**2.0)

    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)

    s_x = r_s*np.cos(lat_rad)*np.cos(lon_rad)
    s_y = - r_s*np.sin(lat_rad)
    s_z = r_s*np.cos(lat_rad)*np.sin(lon_rad)

    lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
    
    return lat, lon


class GeoNEXL1B(object):
    '''
    Get information on L1G data directory, available tiles, years, and files
        file lists are locally cached to future reading as retrieving file lists
        can be time consuming.
    Args:
        data_directory: directory of the L1G product
        sensor: (G16,G17,H8)
    '''
    def __init__(self, data_directory, sensor="G16"):
        self.data_directory = data_directory
        self.sensor = sensor
        self.sat = os.path.basename(os.path.dirname(os.path.dirname(data_directory)))

    def hours(self):
        return list(range(0,24))

    def files(self, year=None, dayofyear=None, hour=None):
        '''
        Args:
            tile (optional): Tile from GeoNEX grid
            year (optional): Year of files to get
            dayofyear (optional): Day of year
        Returns:
            pd.DataFrame of filelist with year, dayofyear, hour, minute, band, file
        '''
        if year == None:
            year = '*'
        else:
            year = str(year)
        if dayofyear == None:
            dayofyear = '*'
        else:
            dayofyear = '%03i' % dayofyear
        if hour == None:
            hour = '*'
        else:
            hour = '%02i' % hour
            
            
        file_pattern = os.path.join(self.data_directory, '%s/%s/%s/*.nc' % (year, dayofyear, hour))
        files = glob.glob(file_pattern)
        fileinfo = []
        for f in files:
            fl = os.path.basename(f).split('_s')[-1]
            y = int(fl[:4])
            doy = int(fl[4:7])
            hour = int(fl[7:9])
            minute = int(fl[9:11])
            b = int(os.path.basename(f).split("M6C")[-1][:2])
            
            fileinfo.append(dict(year=y, dayofyear=doy, hour=hour, minute=minute, band=b, file=f))
        fileinfo = pd.DataFrame(fileinfo)

        return fileinfo
    
    

class L1GFile(object):
    '''
    Reads a single L1B file at a common resolution. Channels are bilinearly interpolated to the defined resolution.
    Args:
        file: Filepath to L1b
        bands (optional): List of bands, default=list(range(1,17))
        resolution_km (optional): Resolution in km for common grid, default=2
    '''
    def __init__(self, files, bands=list((range(1,7))), resolution_km=2.):
        self.files = files
        self.bands = bands
        self.resolution_size = 5424 # 2km
        self.reflective_bands = list(range(1,7))
        self.emissive_bands = list(range(7,17))
        self.lat_lon = None
        

    def load(self, file, band):
        ds = xr.open_dataset(file)
        Rad = np.flipud(ndimage.interpolation.zoom(ds.Rad.values, self.resolution_size/ds.Rad.values.shape[0], order=1))
        
        if self.lat_lon == None:
            lat_rad_1d = ndimage.interpolation.zoom(ds.x.values, self.resolution_size/ds.Rad.values.shape[0], order=1)
            lon_rad_1d = np.flip(ndimage.interpolation.zoom(ds.y.values, self.resolution_size/ds.Rad.values.shape[0], order=1))
            self.lat_lon = GOES_lat_lon(ds, lat_rad_1d, lon_rad_1d)
        
        lat, lon = self.lat_lon[0], self.lat_lon[1]
        Rad = xr.DataArray(data=np.expand_dims(Rad, 0),
                                  dims=["band", "x", "y"],
                                  coords=dict(lat=(["x","y"], lat),
                                              lon=(["x","y"], lon),
                                              band=np.array([band])))
        return Rad
    
    def solar(self):
        fp = SD(self.file, SDC.READ)
        sa = fp.select('Solar_Azimuth').get()[:]
        sz = fp.select('Solar_Zenith').get()[:]
        if sa.shape[0] != self.resolution_size:
            sa = ndimage.interpolation.zoom(sa, self.resolution_size/sa.shape[0], order=1)
            sz = ndimage.interpolation.zoom(sz, self.resolution_size/sz.shape[0], order=1)
        return sa*0.01, sz*0.01
    
    def load_xarray(self, time_stamp):
        Rad_data = [self.load(self.files[self.files.band==b].file.values[0], b) for b in self.bands]
        Rad_data = xr.concat(Rad_data, dim="band")
        Rad_data = Rad_data.expand_dims({"time": list([time_stamp])})
        return xr.Dataset({'L1': Rad_data})
        