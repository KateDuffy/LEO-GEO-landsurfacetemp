from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

from inference import *
import itertools



'''
Perform NEXAI-LST inference on L1G data

Arguments
----------
model_path: str
    Directory location of saved Pytorch model, checkpoint.pth.tar
year: int
    Year of observation to perform inference. Defaults to 2020.
days: int
    Days of year perform inference. Defaults to range(1,366)

Returns
----------
output: torch.nn.Module
    MAIACEmulatorCNN

'''    

model_path = '/nobackupp13/kmduffy1/cross_sensor_training/models/mod11a1/L1G_terra_b7to16_128h_2019/'
year = 2020
days = list(range(1,366))




days_split = [list(f) for f in np.array_split(np.array(days), 4)]
tasks = list(itertools.product(tiles, days_split))
tile, doys = tasks[rank][0],tasks[rank][1]

for d in tqdm.tqdm(doys):
    inference_GEO(model_path, tile, year, d)
