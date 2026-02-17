using MonteCarloX
using Test
using Random

@testset "MonteCarloX.jl" begin
    # utilities
    include("test_utils.jl")
    run_utils_testsets()

    # random number generators
    include("test_rng.jl")
    run_rng_testsets()

    # event handlers
    include("test_event_handler.jl")
    run_event_handler_testsets()

    # measurements
    include("test_measurements.jl")
    run_measurements_testsets()

    # equilibrium / Metropolis
    include("test_metropolis.jl")
    run_metropolis_testsets()

end
