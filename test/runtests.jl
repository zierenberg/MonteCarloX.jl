using MonteCarloX
using Test
using Random

@testset "MonteCarloX.jl" begin
  # Write your own tests here.
  @test Metropolis.update(x->exp(-1*x),1,2,MersenneTwister(1000))==true
end
