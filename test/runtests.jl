using MonteCarloX
using Test
using Random

test_verbose = false

function check(cond::Bool, message::String)
    if test_verbose
        if cond
            printstyled(message; color = :green)
        else
            printstyled(message; color = :red)
        end
    end
    return cond
end

@testset "MonteCarloX.jl" begin
    include("test_binned_objects.jl")
    include("test_checkpointing.jl")
    include("test_ensembles.jl")
    include("test_event_handler.jl")
    include("test_kinetic_monte_carlo.jl")
    include("test_measurements.jl")
    include("test_message_backend.jl")
    include("test_metropolis.jl")
    include("test_multicanonical.jl")
    include("test_parallel_ensembles.jl")
    include("test_rng.jl")
    include("test_utils.jl")
    include("test_wang_landau.jl")
end
