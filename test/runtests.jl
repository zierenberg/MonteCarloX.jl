using MonteCarloX
using Test
using Random
using HypothesisTests
using Distributions
import Distributions.pdf
import Distributions.cdf

@testset "MonteCarloX.jl" begin
    # basics and helper
    include("test_utils.jl")
    @test test_histogram_set_get()
    @test test_log_sum()
    @test test_binary_search()

    include("test_rng.jl")
    @test test_rng_mutable()

    include("test_event_handler.jl")
    @test test_event_handler_rate("ListEventRateSimple")
    @test test_event_handler_rate("ListEventRateActiveMask")

    # equilibrium
    include("test_equilibrium.jl")
    @test test_sweep_random_element()
    @test test_unimodal_metropolis()
    @test test_2D_unimodal_metropolis()
    @test test_unimodal_sweep()

    include("test_ising.jl")
    @test test_ising_reweighting()
    @test test_ising_metropolis()
    @test test_ising_cluster()

    # non-equilibrium
    include("test_inhomogenous_poisson.jl")
    @test test_poisson_single()
    @test test_poisson_constant()
    @test test_poisson_sin_wave()

    # test_kinetic_monte_carlo.jl
    include("test_kinetic_monte_carlo.jl")
    @test test_kmc_next()
    @test test_kmc_advance()
    include("test_gillespie.jl")
    @test test_gillespie()

end
