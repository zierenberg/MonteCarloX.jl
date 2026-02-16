using MonteCarloX
using Test
using Random

@testset "MonteCarloX.jl" begin
    # basics and helper
    include("test_utils.jl")
    @test test_histogram_set_get()
    @test test_log_sum()
    @test test_binary_search()

    # println("rng")
    include("test_rng.jl")
    @test test_rng_mutable()

    # println("event handler")
    include("test_event_handler.jl")
    @test test_event_handler_rate("ListEventRateSimple")
    @test test_event_handler_rate("ListEventRateActiveMask")

    # equilibrium
    # println("equilibrium - Metropolis")
    include("test_metropolis.jl")
    @test test_metropolis_1d_gaussian()
    @test test_metropolis_2d_gaussian()
    @test test_metropolis_acceptance_tracking()
    @test test_metropolis_temperature_effects()
    @test test_metropolis_proposal_invariance()

    # non-equilibrium
    # test_kinetic_monte_carlo.jl
    # include("test_kinetic_monte_carlo.jl")
    # @test test_kmc_next()
    # @test test_kmc_advance()

end
