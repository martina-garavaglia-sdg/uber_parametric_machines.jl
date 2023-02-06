module uber_parametric_machines

using DataFrames, DelimitedFiles, CSV, StatsBase, Flux

export load_data, grouped_data, grouped_data, make_histograms, make_6_months_film, standardize_data, x_y_splitting

include("preprocessing_data.jl")


end
