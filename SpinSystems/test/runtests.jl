using Test
using Random
using MonteCarloX
using SpinSystems

@testset "SpinSystems" begin
    include("test_ising.jl")
    include("test_blume_capel.jl")
end
