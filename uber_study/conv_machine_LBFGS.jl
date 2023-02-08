using DataFrames, DelimitedFiles, CSV
using Flux
using Flux: @epochs, train!
using Flux.Data: DataLoader
using Plots
using ParametricMachinesDemos
using Optim
using FluxOptTools
using uber_parametric_machines

data_apr, data_may, data_jun, data_jul, data_aug, data_sep = load_data();

film = make_6_months_film(data_apr, data_may, data_jun, data_jul, data_aug, data_sep);
st_film = standardize_data(film);

x,y = x_y_splitting(st_film);

x_lbfgs = x[:,:,1:50,:,:]
y_lbfgs = x[:,:,1:50,:,:]

data = DataLoader((x_lbfgs, y_lbfgs));


# (39, 39, 4367, 1, 1), 4367 = 6 months minus 1 day
# Dimensions
dimensions = [1,2,4,8];

machine = ConvMachine(dimensions, sigmoid; pad=(1,1,1,1,10,0)); #tanh, 5

model = Flux.Chain(machine, Conv((1,1,1), sum(dimensions) => 1)) |> f64;

model = cpu(model);


loss() = Flux.Losses.mse(model(x_lbfgs), y_lbfgs);

params = Flux.params(model);


# LBFGS
lossfun, gradfun, fg!, p0 = optfuns(loss, params)
res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=10, store_trace=true))

best_params_PM = res.minimizer
#copy flattened optimized params 
copy!(params, best_params_PM)

Flux.loadparams!(model, params)
