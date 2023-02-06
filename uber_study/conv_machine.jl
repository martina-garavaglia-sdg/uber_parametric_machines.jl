using DataFrames, DelimitedFiles, CSV
using Flux
using Flux: @epochs, train!
using Flux.Data: DataLoader
using Plots
using ParametricMachinesDemos
using uber_parametric_machines

data_apr, data_may, data_jun, data_jul, data_aug, data_sep = load_data();

film = make_6_months_film(data_apr, data_may, data_jun, data_jul, data_aug, data_sep);
st_film = standardize_data(film);

x,y = x_y_splitting(st_film);





