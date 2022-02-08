import datetime as dt
from functools import partial
import itertools
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Proj
import rasterio
import rioxarray as rxr
import scipy.stats
from sklearn.impute import KNNImputer
import tqdm
import xarray as xr

import warnings
warnings.filterwarnings("ignore")


try:
    import geonexl1g
except:
    import data.geonexl1g as geonexl1g


    
dec2bin_numpy = np.vectorize(partial(np.binary_repr, width = 8))
def last_2(string):
    return int(string[-2:])
snip = np.vectorize(last_2)


def read_MOD11A1(file, scan):


    rds = rxr.open_rasterio(file, variable="LST_%s_1km" %scan)
    LST = rds["LST_%s_1km" %scan].values[0,:,:] * 0.02
    LST[LST < 1e-5] = np.nan
    LST = np.expand_dims(LST, 0)
    rds["LST_%s_1km" %scan] = (("band", "y", "x"), LST)
    LST = rds["LST_%s_1km" %scan].rio.reproject("EPSG:4326")[0,:,:]
    lat, lon = LST.y.values, LST.x.values
    rds.close()

    rds = rxr.open_rasterio(file, variable="Emis_31")
    Emis_31 = rds.Emis_31.values[0,:,:] * 0.002
    Emis_31[Emis_31 < 1e-5] = np.nan
    Emis_31 = np.expand_dims(Emis_31, 0)
    rds["Emis_31"] = (("band", "y", "x"), Emis_31)
    Emis_31 = rds.Emis_31.rio.reproject("EPSG:4326")[0,:,:]
    rds.close()

    rds = rxr.open_rasterio(file, variable="Emis_32")
    Emis_32 = rds.Emis_32.values[0,:,:] * 0.002
    Emis_32[Emis_32 < 1e-5] = np.nan
    Emis_32 = np.expand_dims(Emis_32, 0)
    rds["Emis_32"] = (("band", "y", "x"), Emis_32)
    Emis_32 = rds.Emis_32.rio.reproject("EPSG:4326")[0,:,:]
    rds.close()

    rds = rxr.open_rasterio(file, variable="%s_view_angl" %scan)
    angle = rds["%s_view_angl" %scan].rio.reproject("EPSG:4326")[0,:,:]
    angle = np.where(angle.values == 255., np.nan, angle.values) * 1.0 - 65.0
    rds.close()

    rds = rxr.open_rasterio(file, variable="%s_view_time" %scan)
    time = rds["%s_view_time" %scan].rio.reproject("EPSG:4326")[0,:,:]
    longitude, latitude = np.meshgrid(time.x.values, time.y.values)
    time = np.where(time.values == 255., np.nan, time.values)
    time_UTC = time * 0.1 -  (longitude / 15) 
    time_UTC = np.floor(time_UTC) + np.round(time_UTC % 1. * 60, -1) / 60 # hour + nearest 10 minute
    rds.close()

    rds = rxr.open_rasterio(file, variable="QC_%s" %scan)
    QC = rds["QC_%s" %scan].rio.reproject("EPSG:4326")[0,:,:]
    QC_bit = dec2bin_numpy(QC)
    QC_bit2 = snip(QC_bit)
    rds.close()

    ds = xr.Dataset({"mod_LST": (["lat", "lon"], LST),
                    "Emis_31": (["lat", "lon"], Emis_31),
                     "Emis_32": (["lat", "lon"], Emis_32),
                     "mod_view_angl": (["lat", "lon"], angle),
                     "mod_time_UTC": (["lat", "lon"], time_UTC),
                     "mod_QC": (["lat", "lon"], QC_bit2)},
                      coords={"lat": lat, "lon": lon})
    return ds  


def return_tile(lat, lon):
    lat_0 = 60
    lon_0 = -180
    lat_1 = -60
    lon_1 = 180
    h_lons = np.arange(lon_0, lon_1, 6)
    v_lats = np.arange(lat_0, lat_1, -6)
    
    if (len(np.where(lon > h_lons)[0]) > 0.) and (len(np.where(lat < v_lats)[0]) > 0.):
        h = np.where(lon > h_lons)[0][-1]
        v = np.where(lat < v_lats)[0][-1]    
    else:
        h, v = None, None
        
    return h, v
    
    
def get_geonex_tiles_from_latlon_l1g(mod_file, mod_data, time, sensor="G16"):
    lats, lons = mod_data.lat.values,  mod_data.lon.values
    h_ll, v_ll = return_tile(np.nanmin(lats), np.nanmin(lons)) # ll corner of modis tile
    h_ur, v_ur = return_tile(np.nanmax(lats), np.nanmax(lons)) # ur corner of modis tile
    
    
    concat_tiles = []
    if len(list(filter(None, [h_ll, v_ll, h_ur, v_ur]))) == 4.:
        num_tiles = len(list(itertools.product(np.arange(h_ll, h_ur+1), np.arange(v_ur, v_ll+1))))
        for x in itertools.product(np.arange(h_ll, h_ur+1), np.arange(v_ur, v_ll+1)):
            mod_tile = 'h%02iv%02i' % (x[0], x[1])
            
            if sensor == "G16":
                directory = '/nex/datapool/geonex/public/GOES16/GEONEX-L1G/'
            elif sensor == "H8":
                directory = '/nex/datapool/geonex/public/HIMAWARI8/GEONEX-L1G/'
            geo = geonexl1g.GeoNEXL1G(directory, sensor)
            
            if (mod_tile in geo.tiles()):
                year = int(mod_file.split("/")[-2].split(".")[0])
                month = int(mod_file.split("/")[-2].split(".")[1])
                day = int(mod_file.split("/")[-2].split(".")[2])
                hour, minute = int(np.floor(time)), int(np.round(time % 1. * 60, -1))
                hour = hour-24 if hour>24 else hour+24 if hour<0 else hour
                d = dt.datetime(year, month, day)
                files = geo.files(tile=mod_tile, year=year, dayofyear=d.timetuple().tm_yday)
                if len(files) > 0:
                    files = files[files.hour == hour]
                    files = files[files.minute == minute]  
                    if len(files) > 0.:
                        geo_data = geonexl1g.L1GFile(files['file'].values[0], resolution_km=2.).load_xarray()
                        concat_tiles.append(geo_data)
        
        if len(concat_tiles) > 0:
            for i, tile in enumerate(concat_tiles):
                if i == 0:
                    concat = concat_tiles[0]
                else:
                    concat = concat.merge(tile)
            concat = concat.rename({"zenith":"L1_zenith", "azimuth":"L1_azimuth"})
            return concat


def crop_geo(mod_data, geo_data):
    
    lon_ll, lat_ll = mod_data.lon[0], mod_data.lat[-1]
    lon_ur, lat_ur = mod_data.lon[-1], mod_data.lat[0]

    cropped_geo = geo_data.where(geo_data.lat > lat_ll)
    cropped_geo = cropped_geo.where(cropped_geo.lon > lon_ll)
    cropped_geo = cropped_geo.where(cropped_geo.lat < lat_ur)
    cropped_geo = cropped_geo.where(cropped_geo.lon < lon_ur)
    cropped_geo = cropped_geo.dropna('lat', 'all').dropna('lon', 'all')
        
    return cropped_geo    
    
def closest_nonzero(lst, start_index):
    nonzeros = [(i, x) for i, x in enumerate(lst) if x != 0]
    sorted_nonzeros = sorted(nonzeros, key=lambda x: abs(x[0] - start_index))
    return sorted_nonzeros[0][1]


def time_composite(geo_tiles, times):
    
    if (np.nanmax(times[times == times]) - np.nanmin(times[times == times])) < 1: #then don't do composite
        times[:] = np.nanmean(times) 
        
    mode, count = scipy.stats.mode(times, axis=0)
    mode = mode.flatten()
    mode[mode != mode] = 0.
    emptycols = np.nansum(times, axis=0) == 0.

    for col in range(times.shape[1]):
        if emptycols[col]:
            val = closest_nonzero(list(mode), col)
            times[:,col] =  val
    
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=1)
    times_fill = imputer.fit_transform(times)
    times_unique = np.unique(times[times==times])
    x = np.subtract.outer(times_fill, times_unique)
    y = np.argmin(abs(x), axis=-1)
    times_fill = times_unique[y]
    
    L1_zenith = np.zeros_like(geo_tiles[0].L1_zenith.values)
    for i, t in enumerate(times_unique):
        L1_zenith = np.where(times_fill == t, geo_tiles[i].L1_zenith.values, L1_zenith)

    L1_azimuth = np.zeros_like(geo_tiles[0].L1_azimuth.values)
    for i, t in enumerate(times_unique):
        L1_azimuth = np.where(times_fill == t, geo_tiles[i].L1_azimuth.values, L1_zenith)

    L1 = np.zeros_like(geo_tiles[0].L1.values)
    for i, t in enumerate(times_unique):
        times_fill_L1 = np.repeat(times_fill[...,np.newaxis], L1.shape[-1], axis=-1)
        L1 = np.where(times_fill_L1 == t, geo_tiles[i].L1.values, L1)   
        
    out = geo_tiles[0].copy()
    out["L1"] = (("lat", "lon", "band"), L1)
    out["L1_zenith"] = (("lat", "lon"), L1_zenith)
    out["L1_azimuth"] = (("lat", "lon"), L1_azimuth)
    
    return out


def get_elevation(ds, elevation):
    
        y1, y2 = np.min(ds.lat.values)-1, np.max(ds.lat.values)+1
        x1, x2 = np.min(ds.lon.values)-1, np.max(ds.lon.values)+1
        elevation_patch =  elevation.sel(y=slice(y1, y2)).sel(x=slice(x1, x2))

        elevation_patch.load()
        elevation_patch = elevation_patch.interpolate_na(dim="x", method="linear")
        elevation_patch = elevation_patch.interp(y=ds.lat, x=ds.lon)
        return elevation_patch.z.values/8518.0
    
    
def check_inside_scan(ds):
    if (np.sum(ds.mod_LST < 0) == 0.):
        return True
    else:
        return False

    
def check_valid_LST(ds):
    if (np.sum(ds.mod_LST == ds.mod_LST) > 500.):
        return True
    else:
        return False
    
    
def check_valid_L1(ds):
    if (np.sum(ds.L1.values != ds.L1.values) == 0) & (np.nansum(ds.L1.sel(band=slice(7,16)).values < 1e-6) < 1):
        return True
    else:
        return False
    
    
def make_patches(ds, mod_file, save_dir, size=63, stride=64):

    i_dim = ds.mod_LST.shape[0]
    j_dim = ds.mod_LST.shape[1]

    for i in np.arange(0, i_dim-size, stride):
        for j in np.arange(0, j_dim-size, stride):

            lats, lons = ds.lat, ds.lon
            ds_patch = ds.sel(lat=slice(lats[i], lats[i+size])).sel(lon=slice(lons[j], lons[j+size]))
            
            in_scan = check_inside_scan(ds_patch)
            valid_LST = check_valid_LST(ds_patch)
            valid_L1 = check_valid_L1(ds_patch)
            
            # keep 1/3 of patches with all cloud/water
            if not valid_L1:
                if np.random.uniform(low=0, high=1, size=1) > 2/3:
                    valid_L1 == True

            if in_scan & valid_LST & valid_L1:

                fname = mod_file.split("/")[-1].split(".hdf")[0] + "_lat_%s_lon_%s.nc" %(np.round(lats[i].values,2), np.round(lons[j].values,2))
                ds_patch.to_netcdf(save_dir + fname)      
                ds_patch.close()


    


def prepare_tiles(date, LEO_sensor, GEO_sensor):

    elevation = glob.glob("/nobackupp13/kmduffy1/SRTM30/*")
    elevation_ds = xr.open_mfdataset(elevation, combine="by_coords")

    year = date.split(".")[0]
    tile_save_dir = "/nobackupp13/kmduffy1/cross_sensor_training/tiles_%s_%s/%s/" %(LEO_sensor, GEO_sensor, year)
    patch_save_dir = "/nobackupp13/kmduffy1/cross_sensor_training/patches_%s_%s/%s/" %(LEO_sensor, GEO_sensor, year)


    if LEO_sensor == "terra":
        mod11 = glob.glob("/nex/datapool/modis/MOD11A1.006/%s/*" %date)
    elif LEO_sensor == "aqua":
        mod11 = glob.glob("/nex/datapool/modis/MYD11A1.006/%s/*" %date)
    
    
    
    for mod_file in tqdm.tqdm(mod11):
        for scan in ["Day", "Night"]:
            
            try:

                mod_data = read_MOD11A1(mod_file, scan)

                if (mod_data.lat.values.mean() > -60) & (mod_data.lat.values.mean() < 60):


                    times = np.round(mod_data.mod_time_UTC.values,3)
                    geo_tiles = [get_geonex_tiles_from_latlon_l1g(mod_file, mod_data, time, GEO_sensor) for time in np.unique(times[times == times])]

                    geo_tiles = list(filter(None, geo_tiles))
                    if (len(geo_tiles)> 0) & (len(geo_tiles) == len(np.unique(times[times == times]))):

                        cropped_geo_tiles = [crop_geo(mod_data, tile) for tile in geo_tiles]

                        mod_interp = mod_data.interp(dict(lat=cropped_geo_tiles[0].lat, lon=cropped_geo_tiles[0].lon), method="nearest")
                        mod_interp_times = mod_data.interp(dict(lat=cropped_geo_tiles[0].lat, lon=cropped_geo_tiles[0].lon), method="nearest")
                        times = np.round(mod_interp_times.mod_time_UTC.values,3)

                        geo_data = time_composite(cropped_geo_tiles, times)
                        pair = geo_data.merge(mod_interp)

                        elevation = get_elevation(pair, elevation_ds)
                        pair["elevation"] = (("lat", "lon"), elevation)   

                        fname = mod_file.split("/")[-1].split(".hdf")[0] + "_" + scan + ".nc"
                        pair.to_netcdf(tile_save_dir + fname) 
                        print("saved ", tile_save_dir + fname)


                        if year == "2019":
                            make_patches(pair, mod_file, patch_save_dir, size=63)
                            pair.close()
     
                        
            except:
                print("error on ", scan, mod_file)

                
                
if __name__ == "__main__":
    prepare_tiles("2019.09.11", "terra", "G16")
    
    