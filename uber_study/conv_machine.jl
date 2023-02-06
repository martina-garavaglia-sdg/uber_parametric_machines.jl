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

data = DataLoader((x, y));


# (39, 39, 4367, 1, 1)
# Dimensions
dimensions = [1,2,4,8];

machine = ConvMachine(dimensions, sigmoid; pad=(1,1,1,1,10,0)); #tanh, 5

model = Flux.Chain(machine, Conv((1,1,1), sum(dimensions) => 1));

model = cpu(model);

opt = ADAM(0.01);

params = Flux.params(model);

# Loss function
loss(x,y) = Flux.Losses.mse(model(x), y); #mse

# Training and plotting
epochs = Int64[]
loss_on_train = Float64[]
best_params = Float32[]

for epoch in 1:1

    # Train
    Flux.train!(loss, params, data, opt)

    # Saving loss for visualization
    push!(epochs, epoch)
    push!(loss_on_train, loss(x, y))
    @show loss(x, y)

    # Saving the best parameters
    # if epoch > 1
    #     if is_best(loss_on_train[epoch-1], loss_on_train[epoch])
    #         best_params = params
    #     end
    # end
end

# Extract and add new trained parameters
# if isempty(best_params)
#     best_params = params
# end

# Flux.loadparams!(model, best_params);


############################################
############# Visualization ################
############################################

# Loss
plot(epochs, loss_on_train, c=:blue, lw=2, ylims = (0,1));
title!("Convolutional machine");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("visualization/losses/loss_conv_machine.png");


m = model(x);
# Heatmap for the last hour 
heatmap(m[:,:,4360], color=:thermal, clims=(0, 1))

