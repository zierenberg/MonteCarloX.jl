using Test
using Random
using MonteCarloX
using MCXSpins

@testset "MCXSpins" begin
    include("test_ising.jl")
    include("test_blume_capel.jl")
    include("test_nonreciprocal.jl")
end
