using Test
using Random
using MonteCarloX
using MCXSpins
using StaticArrays: SVector

@testset "MCXSpins" begin
    include("test_spin_types.jl")
    include("test_geometries.jl")
    include("test_spin_system.jl")
    include("test_nonreciprocal.jl")
    include("test_structure_factor.jl")
    include("test_correlation.jl")
    include("test_exact_solutions.jl")
end
