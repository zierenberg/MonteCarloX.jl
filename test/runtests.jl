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

    # weights
    include("test_weights.jl")
    run_weights_testsets()

    # generalized ensembles
    include("test_multicanonical.jl")
    run_multicanonical_testsets()
    include("test_wang_landau.jl")
    run_wang_landau_testsets()
    include("test_parallel_ensembles.jl")
    run_parallel_ensembles_testsets()

    # non-equilibrium / kinetic Monte Carlo
    include("test_kinetic_monte_carlo.jl")
    run_kinetic_monte_carlo_testsets()

end
