module uber_parametric_machines

using DataFrames, DelimitedFiles, CSV, StatsBase, Flux, Dates

export load_data, grouped_data, grouped_data, make_histograms, make_6_months_film, standardize_data, x_y_splitting

include("processing_data.jl")


end
