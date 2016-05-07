# download and install the BenchmarkTools Julia package
# Pkg.clone("https://github.com/JuliaCI/BenchmarkTools.jl")

# load the BenchmarkTools module into the current Julia session
using BenchmarkTools

# `@benchmark` defines, tunes, and runs the expression you provide it
trial_results = @benchmark sin(1)

# print the time value of the median estimate to STDOUT
#map(println, trial_results.times)
println(time(median(trial_results)))