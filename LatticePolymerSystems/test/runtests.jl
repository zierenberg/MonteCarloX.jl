using Test

@testset "LatticePolymerSystems" begin
    include("test_lattice_polymer.jl")
    include("test_mc_updates.jl")
end
