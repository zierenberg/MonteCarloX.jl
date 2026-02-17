using Test
using Random
using MonteCarloX
using SpinSystems

@testset "Ising bookkeeping" begin
    sys = Ising([2, 2], J=1)

    @test energy(sys) == -4
    @test energy(sys; full=true) == -4
    @test magnetization(sys) == 4

    ΔE = delta_energy(sys, 1)
    @test ΔE == 4

    modify!(sys, 1, ΔE)
    @test energy(sys) == 0
    @test energy(sys; full=true) == energy(sys)
    @test magnetization(sys) == 2
    @test sys.spins[1] == -1
end

@testset "Ising Metropolis integration" begin
    rng = MersenneTwister(2024)
    sys = Ising([4, 4])
    init!(sys, :random, rng=rng)
    alg = Metropolis(rng, β=0.4)

    @test acceptance_rate(alg) == 0.0
    energy_cache_ok() = energy(sys) == energy(sys; full=true)

    spin_flip!(sys, alg)
    @test alg.steps == 1
    @test 0.0 <= acceptance_rate(alg) <= 1.0
    @test energy_cache_ok()

    for _ in 1:50
        spin_flip!(sys, alg)
    end

    @test energy_cache_ok()
    @test 0.0 <= acceptance_rate(alg) <= 1.0
end
