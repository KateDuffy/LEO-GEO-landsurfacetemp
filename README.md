# Multi-sensor machine learning to retrieve high spatiotemporal resolution land surface temperature

Geostationary (GEO) sensors provide near-continuous, continental-scale observations which can better capture the diurnal variability of land surface temperature (LST) than intermittent observations from low-earth orbit (LEO) sensors. This repository contains code to pair co-located, co-temporal observations from LEO and GEO satellites, train a convolutional neural network to predict MODIS-like LST from GEO thermal emissive bands, and perform inference on GEO observations. The resulting NASA Earth eXchange Artificial Intelligence LST (NEXAI-LST) achieved a mean absolute error of 1.73 K relative to the target LEO product and improves on both spatial and temporal resolution [2km, 10 minute] compared to the GEO full disk standard product [10km, hourly].

## Install

Clone repository
```
git clone https://github.com/KateDuffy/LEO-GEO-landsurfacetemp.git
```

Create conda environment
```
conda env create -f environment.yml
conda activate lst
```

Install appropriate PyTorch distribution (See: https://pytorch.org/)
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Datasets
This project uses datasets from the NASA Earth eXchange (NEX). One year of NEXAI-LST data is available through [https://data.nas.nasa.gov](https://data.nas.nasa.gov/geonex/geonexdata/ML/nexai-lst/). This data is considered provisional and is provided to facilitate data exploration and further studies.

## Generating training data
Pairs of LEO LST and GEO Level 1 data comprise training data. Co-incident training samples are prepared in `prepare_training_data_L1G.py`.
```
from data.prepare_training_data_L1G import prepare_tiles
prepare_tiles(date="2019.01.01", LEO_sensor="terra", GEO_sensor"G16")
```

## Training
The file `train_emulator` contains code to train a convolutional neural network implemented in PyTorch.

## Inference
To predict LST for new geostationary data, `inference.py` contains code to perform inference on GOES-16 (G16) or Himawari-8 (H8) for a given GeoNEX tile, year, and day of year.

```
from inference import inference_GEO
inference_GEO([model_path], [save_directory], tile='h08v01', year=2020, doy=1, sensor='G16')
```
