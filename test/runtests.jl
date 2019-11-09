using MonteCarloX
using Test
using Random
using HypothesisTests
using Distributions
import Distributions.pdf
import Distributions.cdf

@testset "MonteCarloX.jl" begin
  # Write your own tests here.
  @test Metropolis.update(x->exp(-1*x),1,2,MersenneTwister(1000))==true

  include("../examples/inhomogeneous_poisson_tests.jl")
  @test test_poisson_single()
  @test test_poisson_constant()
  @test test_poisson_sin_wave()
end
