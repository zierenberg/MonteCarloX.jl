using Test
using MonteCarloX
using MCXSpins

@testset "Exact 2D Ising logDOS API" begin
    binned = logdos_exact_ising2D(L=8)
    @test binned isa BinnedObject
    @test binned[-128] == log(2)
    @test binned[0] ≈ 42.41274640460084
    @test isnan(binned[-124])

    vec = logdos_exact_ising2D(8; format=:vector)
    @test vec isa Vector{Tuple{Int,Float64}}
    @test first(vec) == (-128, log(2))
    @test last(vec) == (128, log(2))

    dict = logdos_exact_ising2D(8; format=:dict)
    @test dict[0] ≈ 42.41274640460084

    @test_throws ErrorException logdos_exact_ising2D(L=10)
    @test_throws ErrorException logdos_exact_ising2D(8; format=:unknown)
end
