using MonteCarloX
using Test
using Random
using HypothesisTests
using Distributions
import Distributions.pdf
import Distributions.cdf

@testset "MonteCarloX.jl" begin
  # Write your own tests here.
  #include("test_histograms.jl")

  include("test_equilibrium.jl")
  @test test_sweep_random_element()
  @test test_unimodal_metropolis()
  @test test_2D_unimodal_metropolis()
  @test test_unimodal_sweep()

  include("test_inhomogenous_poisson.jl")
  @test test_poisson_single()
  @test test_poisson_constant()
  @test test_poisson_sin_wave()

  include("test_gillespie.jl")
  #@test test_gillespie()

  include("test_ising.jl")
  @test test_ising_reweighting()
  @test test_ising_metropolis()
  @test test_ising_cluster()
  
end
