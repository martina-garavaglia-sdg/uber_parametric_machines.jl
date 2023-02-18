using DataFrames, DelimitedFiles, CSV, Dates #Query
using StatsBase, Flux

# Loading
function load_data()
    data_apr = DataFrame(CSV.File("data/uber-raw-data-apr14.csv"));
    data_may = DataFrame(CSV.File("data/uber-raw-data-may14.csv"));
    data_jun = DataFrame(CSV.File("data/uber-raw-data-jun14.csv"));
    data_jul = DataFrame(CSV.File("data/uber-raw-data-jul14.csv"));
    data_aug = DataFrame(CSV.File("data/uber-raw-data-aug14.csv"));
    data_sep = DataFrame(CSV.File("data/uber-raw-data-sep14.csv"));
    return data_apr, data_may, data_jun, data_jul, data_aug, data_sep
end

data_apr, data_may, data_jun, data_jul, data_aug, data_sep = load_data()




function grouped_data(data::DataFrame)
    d = copy(data)
    d.DateTime = DateTime.(d.DateTime, "mm/dd/yyyy HH:MM:SS")
    d[!,:hours] = hour.(d[!,:DateTime])
    d[!,:days] = day.(d[!,:DateTime]);
    return groupby(d, [:days, :hours]);
end




function make_histograms(data::DataFrame)
    d = copy(data)
    df_grouped = grouped_data(d)

    lat_max = d[!,:Lat]
    lon_max = d[!,:Lon]

    bins_lat = range(minimum(lat_max), maximum(lat_max), length=101)
    bins_lon = range(minimum(lon_max), maximum(lon_max), length=101)

    film = []

    for i in 1:length(df_grouped)
        lat = df_grouped[i][!,:Lat]
        lon = df_grouped[i][!,:Lon]
        H = fit(Histogram, (lat, lon), (bins_lat, bins_lon))

        push!(film, H.weights)
        
    end

    film = cat(film..., dims=3)
    film = Flux.unsqueeze(film, dims=4)
    film = Flux.unsqueeze(film, dims=5)

    return film
end

function make_6_months_film(data1::DataFrame, data2::DataFrame, data3::DataFrame, data4::DataFrame, data5::DataFrame, data6::DataFrame)
    
    film1 = make_histograms(data1)
    film2 = make_histograms(data2)
    film3 = make_histograms(data3)
    film4 = make_histograms(data4)
    film5 = make_histograms(data5)
    film6 = make_histograms(data6)
    
    film = []
    push!(film, film1)
    push!(film, film2)
    push!(film, film3)
    push!(film, film4)
    push!(film, film5)
    push!(film, film6)
    film = cat(film..., dims=3)

    return film
end


function standardize_data(f::Array)

    M = maximum(f)
    m = minimum(f)
    st_data = (f.-m) ./ (M-m)
    
    return st_data, M, m
end


function standardize_data(f::Array, M, m)
    return (f.-m) ./ (M-m)
end
