# uber_parametric_machines

[![Build Status](https://github.com/martina-garavaglia-sdg/uber_parametric_machines.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/martina-garavaglia-sdg/uber_parametric_machines.jl/actions/workflows/CI.yml?query=branch%3Amaster)


The aim of this repository is predicting uber pickups in New York city using parametric machines.
Data have a geospatial and temporal dynamics which suggested that I use a convolutional architecture for this task.

This repository is organized as follows:
- Data preprocessing: data are preprocessed so that they formed a film of spatial grids representing uber pickups activity in different New York areas. Data are then standardized and splitted into train and test set;
- Model initialization: convolutional machine is initialize within their parameters. The optimizers are choosen (Adam and LBFGS) as well as the loss functions (mse and mae). 
- Training
- Results visualization

NB: a more in depth explanation can be found in this [repository](https://github.com/martina-garavaglia-sdg/master_thesis).
