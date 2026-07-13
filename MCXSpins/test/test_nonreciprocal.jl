using Test
using Random
using MonteCarloX
using MCXSpins

@testset "Nonreciprocal constructors" begin
    si = VisionConeIsingSystem([4, 4]; κ=0.3)
    @test si isa SpinSystem
    @test si.spintype isa Spin{1//2}
    @test si.interactions[1] isa PairInteraction          # reciprocal J part
    @test si.interactions[2] isa VisionConeInteraction    # cone extra κ
    @test all(==(Int8(1)), si.spins)
    @test !is_hamiltonian(si)

    sb = VisionConeBlumeCapelSystem([4, 4]; κ=0.3, D=0.5)
    @test sb.spintype isa Spin{1//1}
    @test sb.interactions[3] isa CrystalField
    @test sb.interactions[3].Δ == 0.5
end

@testset "Nonreciprocal initialization" begin
    rng = MersenneTwister(7)
    sb = VisionConeBlumeCapelSystem([4, 4]; κ=0.3, D=0.2)

    init!(sb, :down)
    @test all(==(Int8(-1)), sb.spins)
    @test magnetization(sb) == -16

    init!(sb, :zero)
    @test all(==(Int8(0)), sb.spins)
    @test magnetization(sb) == 0

    init!(sb, :random, rng=rng)
    @test all(s in Int8[-1, 0, 1] for s in sb.spins)
    @test magnetization(sb) == sum(Int, sb.spins)

    # :zero is invalid for the two-state Ising model
    si = VisionConeIsingSystem([4, 4]; κ=0.0)
    @test_throws ErrorException init!(si, :zero)
end

@testset "κ=0 reduces to equilibrium dynamics" begin
    # At κ = 0 the cone term contributes nothing and the pair term is the plain Ising
    # coupling: every local ΔE (probed for every site and target state) and the whole
    # Glauber trajectory under a shared seed must coincide with the equilibrium system.
    # (The cone term stays cache-free by type, so `energy` still refuses — the comparison
    # is on the dynamics and on the Hamiltonian part.)
    rng = MersenneTwister(2026)
    nr = VisionConeIsingSystem([6, 6]; κ=0.0, J=1)
    init!(nr, :random, rng=rng)
    eq = IsingSystem([6, 6]; J=1)
    set_spins!(eq, nr.spins)

    dev = maximum(abs(delta_energy(nr, i, Int8(-nr.spins[i])) -
                      delta_energy(eq, i, Int8(-eq.spins[i]))) for i in 1:36)
    @test dev == 0

    alg_nr = GlauberAlgorithm(MersenneTwister(5); β=0.5)
    alg_eq = GlauberAlgorithm(MersenneTwister(5); β=0.5)
    for _ in 1:50 * 36
        spin_flip!(nr, alg_nr)
        spin_flip!(eq, alg_eq)
    end
    @test nr.spins == eq.spins
    @test acceptance_rate(alg_nr) == acceptance_rate(alg_eq)

    # Blume-Capel: same, including the crystal-field term.
    nb = VisionConeBlumeCapelSystem([6, 6]; κ=0.0, D=0.75, J=1)
    init!(nb, :random, rng=rng)
    bc = BlumeCapelSystem([6, 6]; J=1, D=0.75)
    set_spins!(bc, nb.spins)
    alg_nb = GlauberAlgorithm(MersenneTwister(6); β=0.5)
    alg_bc = GlauberAlgorithm(MersenneTwister(6); β=0.5)
    for _ in 1:50 * 36
        spin_flip!(nb, alg_nb)
        spin_flip!(bc, alg_bc)
    end
    @test nb.spins == bc.spins
    @test hamiltonian_energy(nb) ≈ energy(bc)   # pair + crystal-field parts match (h = 0)
end

@testset "Vision-cone asymmetry" begin
    # 4×4 lattice, site 1: +axis partners (right, up) = (2, 5); −axis (left, down) = (4, 13).
    # Config: +dir sum = −2, −dir sum = +2. The cone term is ΔE = −(s_new − s_old)·κ·fwd
    # with fwd following the CURRENT spin's cone (the reciprocal J part lives in the
    # separate PairInteraction and cancels here: sp + sn = 0).
    sys = VisionConeIsingSystem([4, 4]; κ=0.5, J=1)
    spins = zeros(Int8, 16)
    spins[2] = -1; spins[5] = -1               # +dir sum = −2
    spins[4] = 1; spins[13] = 1                # −dir sum = +2
    vc = sys.interactions[2]

    spins[1] = Int8(1)                          # up spin sees +dir: fwd = −2
    @test MCXSpins.delta(vc, spins, 1, Int8(-1)) == -(-2) * (0.5 * -2) == -2.0

    spins[1] = Int8(-1)                         # down spin sees −dir: fwd = +2
    @test MCXSpins.delta(vc, spins, 1, Int8(1)) == -(2) * (0.5 * 2) == -2.0

    # κ = 0: the cone term vanishes identically — reciprocal limit.
    vc0 = VisionConeIsingSystem([4, 4]; κ=0.0, J=1).interactions[2]
    spins[1] = Int8(1)
    @test MCXSpins.delta(vc0, spins, 1, Int8(-1)) == 0.0
end

@testset "spin_flip! sanity and bookkeeping (Glauber and Metropolis)" begin
    for makealg in (rng -> GlauberAlgorithm(rng; β=0.4), rng -> MetropolisAlgorithm(rng; β=0.4))
        rng = MersenneTwister(11)
        sys = VisionConeBlumeCapelSystem([8, 8]; κ=0.3, D=0.5)
        init!(sys, :random, rng=rng)
        alg = makealg(rng)

        N = length(sys.spins)
        nsweeps = 100
        for _ in 1:nsweeps, _ in 1:N
            spin_flip!(sys, alg)
        end

        @test alg.steps == nsweeps * N
        @test 0.0 <= acceptance_rate(alg) <= 1.0
        @test all(s in Int8[-1, 0, 1] for s in sys.spins)

        # caches stay consistent with a full recompute
        @test magnetization(sys) == sum(Int, sys.spins)
        @test sys.interactions[3].cache.val == sum(s -> Int(s)^2, sys.spins)
        @test isfinite(hamiltonian_energy(sys))
        @test_throws ErrorException energy(sys)                 # no Hamiltonian, by design
        @test correlation_length(sys) >= 0.0
    end
end

@testset "Nonreciprocity shifts ordering (Tc grows with κ)" begin
    # Low-temperature smoke test: at fixed β below the reciprocal Tc, larger κ (which raises Tc)
    # should not destroy order. We check that |m| stays high across κ at a cold temperature.
    function mean_abs_m(κ; β, L=16, warmup=200, samples=200, seed=99)
        rng = MersenneTwister(seed)
        sys = VisionConeIsingSystem([L, L]; κ=κ)
        init!(sys, :up)
        alg = GlauberAlgorithm(rng; β=β)
        N = L * L
        for _ in 1:warmup, _ in 1:N
            spin_flip!(sys, alg)
        end
        acc = 0.0
        for _ in 1:samples
            for _ in 1:N
                spin_flip!(sys, alg)
            end
            acc += abs(magnetization(sys)) / N
        end
        return acc / samples
    end

    β_cold = 1 / 2.0    # T = 2.0 < Tc(κ=0) ≈ 2.269
    @test mean_abs_m(0.0; β=β_cold) > 0.6
    @test mean_abs_m(0.5; β=β_cold) > 0.6
    @test mean_abs_m(1.0; β=β_cold) > 0.6
end
