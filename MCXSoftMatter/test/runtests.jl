using Test
using Random
using MCXSoftMatter

@testset "MCXSoftMatter" begin
    include("test_periodic.jl")
    include("test_potentials.jl")
    include("test_particle_gas.jl")
    include("test_bead_spring_polymer.jl")
    include("test_mc_updates.jl")
end
