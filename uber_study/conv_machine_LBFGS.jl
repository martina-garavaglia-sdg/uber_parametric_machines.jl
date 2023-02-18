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
film_1_month = film[:,:,:,:] 

x_lbfgs = film[:,:,1:24*27,:]
y_lbfgs = film[:,:, 25:24*28,:]
x_lbfgs, max, min = standardize_data(x_lbfgs);
y_lbfgs = standardize_data(y_lbfgs, max, min);

x_lbfgs = x_lbfgs[11:50, 11:50,:,:]
y_lbfgs = y_lbfgs[11:50, 11:50,:,:]

x_lbfgs = Flux.unsqueeze(x_lbfgs, dims=4)
y_lbfgs = Flux.unsqueeze(y_lbfgs, dims=4)

data = DataLoader((x_lbfgs, y_lbfgs));


# (40, 40, 648, 1, 1), 648=24*27
# Dimensions
dimensions = [1,2,4,8];

machine_lbfgs = ConvMachine(dimensions, sigmoid; pad=(1,1,1,1,10,0)); #tanh, 5

model_lbfgs = Flux.Chain(machine_lbfgs, Conv((1,1,1), sum(dimensions) => 1)) |> f64;

model_lbfgs = cpu(model_lbfgs);


loss_lbfgs() = Flux.Losses.mse(model_lbfgs(x_lbfgs), y_lbfgs);

params_lbfgs = Flux.params(model_lbfgs);


# LBFGS
lossfun, gradfun, fg!, p0 = optfuns(loss_lbfgs, params_lbfgs)
res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=500, store_trace=true))

best_params_PM_lbfgs = res.minimizer
#copy flattened optimized params 
copy!(params_lbfgs, best_params_PM_lbfgs)

Flux.loadparams!(model_lbfgs, params_lbfgs)


Flux.Losses.mse(model_lbfgs(x_lbfgs), y_lbfgs)

# Metrics

h = 24
d_test = 26

m_lbfgs = model_lbfgs(x_lbfgs)
pred = m_lbfgs[:,:,h*(d_test-1):h*d_test,:,:]; # prediction day 7
gt = y[:,:,h*(d_test-1):h*d_test,:,:]; # day 7
pred_naive1 = zeros(size(gt));
pred_naive2 = y[:,:,h*(d_test-2):h*(d_test-1),:]; # day 6

##### Model naive 1: comparing my data to zero model usong mse and mae
error_naive1_mse_test_LBFGS = Flux.Losses.mse(pred_naive1, gt)
error_naive1_mae_test_LBFGS = Flux.Losses.mae(pred_naive1, gt)

# Model naive 2: comparing my data to my data the day before
error_naive2_mse_test_LBFGS = Flux.Losses.mse(pred_naive2, gt)
error_naive2_mae_test_LBFGS = Flux.Losses.mae(pred_naive2, gt)

# Test error model
error_test_set_mse_LBFGS = Flux.Losses.mse(pred, gt)
error_test_set_mae_LBFGS = Flux.Losses.mae(pred, gt)



################################################################
######### Average on space, error for every hour ###############
################################################################


# Model Naive 1
error_naive1_hour_test_LBFGS = []
for hour in 1:h
    push!(error_naive1_hour_test_LBFGS, Flux.Losses.mae(zeros(27,27,1,1,1), y_lbfgs[:,:,h*(d_test-1)+hour,:,:])) 
end
error_naive1_hour_test_LBFGS

# Model Naive 2
error_naive2_hour_test_LBFGS = []
for hour in 1:h
    push!(error_naive2_hour_test_LBFGS, Flux.Losses.mae(y_lbfgs[:,:,h*(d_test-1)+hour,:], y_lbfgs[:,:,h*(d_test-1)+hour-1,:,:])) 
end
error_naive2_hour_test_LBFGS

#####

# Test error

error_hour_mae_test_LBFGS = []
for hour in 1:h
    push!(error_hour_mae_test_LBFGS, Flux.Losses.mae(m_lbfgs[:,:,h*(d_test-1)+hour,:,:], y_lbfgs[:,:,h*(d_test-1)+hour,:,:])) 
end
error_hour_mae_test_LBFGS


plot(error_naive1_hour_test_LBFGS, lab="Naive 1")
title!("LBFGS optimizer")
plot!(error_naive2_hour_test_LBFGS, lab="Naive 2")
plot!(error_hour_mae_test_LBFGS, lab="Test error")


