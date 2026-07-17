using MonteCarloX
using Random
using Test

function test_algorithm_steps_interface()
    pass = true

    @eval begin
        struct _DummyAlg <: AbstractAlgorithm end
        @test_throws ArgumentError steps(_DummyAlg())
    end

    alg_metro = MetropolisAlgorithm(Xoshiro(1); β=1.0)
    pass &= check(steps(alg_metro) == 0, "Metropolis steps starts at 0\n")
    accept!(alg_metro, 0.0, 1.0)
    pass &= check(steps(alg_metro) == 1, "Metropolis steps increments\n")

    alg_glauber = GlauberAlgorithm(Xoshiro(2); β=1.0)
    pass &= check(steps(alg_glauber) == 0, "Glauber steps starts at 0\n")
    accept!(alg_glauber, 1.0)                       # 1-arg accept! takes logR directly
    pass &= check(steps(alg_glauber) == 1, "Glauber steps increments\n")

    alg_heat = HeatBathAlgorithm(Xoshiro(3); β=1.0)
    pass &= check(steps(alg_heat) == 0, "HeatBath steps starts at 0\n")
    alg_heat.steps += 1
    pass &= check(steps(alg_heat) == 1, "HeatBath steps readable\n")

    alg_gillespie = Gillespie(Xoshiro(4))
    pass &= check(steps(alg_gillespie) == 0, "Gillespie steps starts at 0\n")

    rx = ParallelTempering([1.0, 0.5]; seed=10, rng=Xoshiro)
    pass &= check(steps(rx) == 0, "ReplicaExchange steps starts at 0\n")
    update!(rx, [-1.0, -0.5])
    pass &= check(steps(rx) >= 0, "ReplicaExchange steps readable\n")

    return pass
end

@testset "Algorithm steps interface" begin
    @test test_algorithm_steps_interface()
end
