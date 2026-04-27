using Test

@testset "LatticePolymerSystems" begin
    include("test_cubic_lattice.jl")
    include("test_lattice_gas.jl")
    include("test_cluster.jl")
    include("test_lattice_polymer.jl")
end
