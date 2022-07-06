from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

from inference import *
import itertools



'''
Perform NEXAI-LST inference on L1G data with multiprocessing

Arguments
----------
model_path: str
    Directory location of saved Pytorch model, checkpoint.pth.tar
save_directory: str
    Directory location to save inferences
year: int
    Year of observation to perform inference. Defaults to 2020.
days: int
    Days of year perform inference. Defaults to range(1,366)
tiles: list
    GeoNEX tile(s) for which to perform inference, such as ['h08v01']
sensor: str
    Name of geostationary sensor. "G16" for GOES-16 and "H8" for Himawari-8
    
''' 

# arguments
model_path = '/nobackupp13/kmduffy1/cross_sensor_training/models/mod11a1/L1G_terra_b7to16_128h_2019/'
save_directory = "/nobackupp13/kmduffy1/NEXAI-LST"
year = 2020
days = list(range(1,366))
tiles = ['h08v01','h09v01','h10v01','h11v01','h12v01']
sensor = "G16"



# divide dates into tasks for multiprocessing
days_split = [list(f) for f in np.array_split(np.array(days), 4)]
tasks = list(itertools.product(tiles, days_split))
tile, doys = tasks[rank][0],tasks[rank][1]

# perform inference
for d in tqdm.tqdm(doys):
    inference_GEO(model_path=model_path, save_directory=save_directory, tile=tile, year=year, doy=d, sensor=sensor)
    
