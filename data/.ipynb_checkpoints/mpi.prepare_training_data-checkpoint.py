from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

from prepare_training_data_L1G import *

LEO_sensor = "terra"
GEO_sensor = "G16"
year = "2020"

if LEO_sensor == "aqua":
    directories = glob.glob("/nex/datapool/modis/MYD11A1.006/%s.*" %year)
if LEO_sensor == "terra":
    directories = glob.glob("/nex/datapool/modis/MOD11A1.006/%s.*" %year)
    
    
dates = [d.split("/")[-1] for d in directories]

prepare_tiles(dates[rank], LEO_sensor, GEO_sensor)

 
