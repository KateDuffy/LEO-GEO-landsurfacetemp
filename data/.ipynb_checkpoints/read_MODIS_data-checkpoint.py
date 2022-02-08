import numpy as np
import tqdm
from pyhdf.SD import SD, SDC 
from geonexl1b import *


def inverse_maps(H,V,size):
    Lat = Lon = np.zeros((size,size))
    R, T = 6371007.181, 1111950
    xmin, ymax = -20015109., 10007555.
    w = T /size
    y=np.array([(ymax-(i+.5)*w-V*T) for i in range(size)] )
    x =np.array([((j+.5)*w + (H)*T + xmin) for j in range(size)])
    for i,yy in enumerate(y):
        for j,xx in enumerate(x):
            ll=yy/R
            Lat[i,j]=ll*180/np.pi
            Lon[i,j]=180/np.pi*(xx/(R*np.cos(ll)))
    return Lat,Lon


def read_MCD18A1(filename):
    file = SD(filename, SDC.READ)
    sds_obj = file.select('DSR') 
    dsr = sds_obj.get()
    tile = os.path.basename(filename).split(".")[2]
    H, V =  int(tile[1:3]), int(tile[4:6])
    lat, lon = inverse_maps(H,V,dsr.shape[-1])
    n_orbits = getattr(file, 'Orbit_amount')
    orbit_time_stamps = getattr(file, 'Orbit_time_stamp').split("\n")[:n_orbits]
    ds = xr.Dataset(data_vars=dict(dsr=(["time", "x", "y"], dsr)),
                    coords=dict(lon=(["x", "y"], lon),
                                lat=(["x", "y"], lat),
                                time=orbit_time_stamps))
    return ds


def read_MOD10_L2(MCD18A1):
    data = []
    for t in MCD18A1.time.values:
        files = "/nobackupp13/kmduffy1/MOD10L2.v061/MOD10_L2.A%s.%s.*.hdf" %(t[:-4], t[-4:])
        file_name = glob.glob(files)[0]
        file = SD(file_name, SDC.READ)
        sds_obj = file.select('NDSI_Snow_Cover')
        data.append(sds_obj.get())
        
    #####################################
    ####### need geolocation data #######
    #####################################
    # lat = 
    # lon = 

#     ds = xr.Dataset(data_vars={snow_cover:(["time", "x", "y"], np.stack(np.array(data)))},
#                     coords=dict(lon=(["x", "y"], lon),
#                                 lat=(["x", "y"], lat),
#                                 time=MCD18A1.time.values))
    
def read_MOD04_3K(MCD18A1):
    pass

def read_GOES_L1B(MCD18A1):
    
    directory = '/nex/datapool/geonex/public/GOES16/NOAA-L1B/ABI-L1b-RadF/'
    geo = GeoNEXL1B(directory)

    L1B = []
    for ts in tqdm.tqdm(MCD18A1.time.values):

        yyyy, doy, hh, mm = int(ts[:4]), int(ts[4:7]), int(ts[7:9]), int(ts[9:11])
        files = geo.files(year=yyyy, dayofyear=doy, hour=hh)
        if len(files) > 0:
            files = files[files.minute == np.round(mm, -1)]
        if len(files) > 0:
            geo_data = L1GFile(files, resolution_km=2.).load_xarray(ts)
            L1B.append(geo_data)

    L1B = xr.concat(L1B, dim="time")
    return L1B
    

