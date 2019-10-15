using MonteCarloX
using Test
using Random

@testset "MonteCarloX.jl" begin
  # Write your own tests here.
  @test Metropolis.update(1,2,x->exp(-1*x),MersenneTwister(1000))==true
end
