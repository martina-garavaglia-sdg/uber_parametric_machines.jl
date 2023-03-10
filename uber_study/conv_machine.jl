using DataFrames, DelimitedFiles, CSV
using Statistics
using Flux
using Flux: @epochs, train!
using Flux.Data: DataLoader
using Plots
using ParametricMachinesDemos
using BSON: @save, @load
using uber_parametric_machines

data_apr, data_may, data_jun, data_jul, data_aug, data_sep = load_data();

film = make_6_months_film(data_apr, data_may, data_jun, data_jul, data_aug, data_sep);

x = film[:,:,1:24*27,:]
y = film[:,:,25:24*28,:]
x, max, min = standardize_data(x);
y = standardize_data(y, max, min);

# x = x[11:50, 11:50,:,:]
# y = y[11:50, 11:50,:,:] # tenendo un 40x40 non riesco neanche a far partire il training

## To change grid size:
x = x[16:45, 16:45, :,:]
y = y[16:45, 16:45, :,:]
##

x = Flux.unsqueeze(x, dims=4)
y = Flux.unsqueeze(y, dims=4)

data = DataLoader((x, y));


# (30, 30, 648, 1, 1), 648 = 24*27
# Dimensions
dimensions = [1,2,4,8];

machine = ConvMachine(dimensions, sigmoid; pad=(1,1,1,1,24,0)); #tanh

model = Flux.Chain(machine, Conv((1,1,1), sum(dimensions) => 1)) |> f64;

model = cpu(model);

opt = ADAM(0.01);

params = Flux.params(model);

# Loss function
loss(x,y) = Flux.Losses.mse(model(x), y); #mse  provo con mae

# Training and plotting
epochs = Int64[]
loss_on = Float64[]
best_params = Float32[]

for epoch in 1:300

    # Train
    Flux.train!(loss, params, data, opt)

    # Saving loss for visualization
    push!(epochs, epoch)
    push!(loss_on, loss(x, y))
    @show loss(x, y)

    # Saving the best parameters
    # if epoch > 1
    #     if is_best(loss_on[epoch-1], loss_on[epoch])
    #         best_params = params
    #     end
    # end
end

# Saving model params in a file
@save "500epochs_model.bson" model


# Extract and add new trained parameters
# if isempty(best_params)
#     best_params = params
# end


# Flux.loadparams!(model, best_params);


############################################
############# Visualization ################
############################################

# Loss
plot(epochs, loss_on, legend=false, lw=2, ylims = (0,1));
title!("Convolutional machine - Loss");
yaxis!("Loss");
xaxis!("Training epochs");
savefig("visualization/1_month/loss_conv_machine.png");


m = model(x);

heatmap(y[:,:,648-8], color=:thermal, clims=(0, 0.5))
heatmap(m[:,:,648-8], color=:thermal, clims=(0, 0.05))
# Heatmap for the last hour 
heatmap(abs.(m[:,:,648-8] - y[:,:,648-8]), color=:thermal, clims=(0, 0.5))


savefig("visualization/1_month/hour_630_prediction.png");

model_prova = Flux.Chain(machine, Conv((1,1,1), sum(dimensions) => 1)) |> f64;
@load "model_params/500epochs_model_30x30.bson" model_prova 

##############################################
################# Metrics ####################
##############################################


function smape_marti(forecast::AbstractArray, real::AbstractArray)
    err =  real .- forecast
    abs_err = abs.(err)
    return mean((2 .* abs_err ./ Base.max.(real .+ forecast, eps())))   # divido per tutto prendo solo i non zero
end

# safety check

h = 24
d_test = 26

# m = model(x)
pred = m[:,:,h*(d_test-1):h*d_test,:,:]; # prediction day 7
gt = y[:,:,h*(d_test-1):h*d_test,:,:]; # day 7
pred_naive1 = zeros(size(gt));
pred_naive2 = y[:,:,h*(d_test-2):h*(d_test-1),:]; # day 6

##### Model naive 1: comparing my data to zero model usong mse and mae
error_naive1_mse_test = Flux.Losses.mse(pred_naive1, gt)
error_naive1_mae_test = Flux.Losses.mae(pred_naive1, gt)
error_naive1_smape_test = smape_marti(pred_naive1, gt)

# Model naive 2: comparing my data to my data the day before
error_naive2_mse_test = Flux.Losses.mse(pred_naive2, gt)
error_naive2_mae_test = Flux.Losses.mae(pred_naive2, gt)
error_naive2_smape_test = smape_marti(pred_naive2, gt)


# Test error model
error_test_set_mse = Flux.Losses.mse(pred, gt)
error_test_set_mae = Flux.Losses.mae(pred, gt)
error_test_set_smape = smape_marti(pred, gt)



################################################################
######### Average on space, error for every hour ###############
################################################################


# Model Naive 1
error_naive1_hour_test = []
for hour in 1:h
    push!(error_naive1_hour_test, Flux.Losses.mse(zeros(30,30,1,1,1), y[:,:,h*(d_test-1)+hour,:,:])) 
end
error_naive1_hour_test

# Model Naive 2
error_naive2_hour_test = []
for hour in 1:h
    push!(error_naive2_hour_test, Flux.Losses.mse(y[:,:,h*(d_test-1)+hour,:], y[:,:,h*(d_test-1)+hour-1,:,:])) 
end
error_naive2_hour_test

#####

# Test error

error_hour_mae_test = []
for hour in 1:h
    push!(error_hour_mae_test, Flux.Losses.mse(m[:,:,h*(d_test-1)+hour,:,:], y[:,:,h*(d_test-1)+hour,:,:])) 
end
error_hour_mae_test


plot(error_naive1_hour_test, lab="Naive 1")
plot!(error_naive2_hour_test, lab="Naive 2")
plot!(error_hour_mae_test, lab="Test error")
savefig("visualization/1_month/naive_test_errors_mse.png");



###################################################################
############# Average on time, error for every cell ###############
###################################################################

# Model naive 1

error_naive1_cell_test = []

for i in 1:40
    for j in 1:40
        push!(error_naive1_cell_test, Flux.Losses.mae(zeros(size(y[i,j,(d_test-1)*h:d_test*h,:,:])), y[i,j,(d_test-1)*h:d_test*h,:,:]))
    end
end
error_naive1_cell_test

# Model naive 2

error_naive2_cell_test = []

for i in 1:40
    for j in 1:40
        push!(error_naive2_cell_test, Flux.Losses.mae(y[i,j,h*(d_test-2):h*(d_test-1),:], y[i,j,h*(d_test-1):h*d_test,:,:]))
    end
end
error_naive2_cell_test

# Test error model

error_cell_mae_test = []

for i in 1:40
    for j in 1:40
        push!(error_cell_mae_test, Flux.Losses.mae(m[i,j,h*(d_test-1):h*d_test,:], y[i,j,h*(d_test-1):h*d_test,:,:]))
    end
end
error_cell_mae_test
