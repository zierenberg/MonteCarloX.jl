using Test
using Random
using MonteCarloX
using MCXSpins

@testset "Nonreciprocal constructors" begin
    si = NonreciprocalIsing([4, 4]; κ=0.3)
    @test si isa SpinSystem
    @test si.model isa IsingModel
    @test si.nr isa VisionCone
    @test si.topo isa LatticeTopology
    @test all(==(Int8(1)), si.spins)

    sb = NonreciprocalBlumeCapel([4, 4]; κ=0.3, D=0.5)
    @test sb.model isa BlumeCapelModel
    @test sb.model.Δ == 0.5

    # General path and Reciprocal identity
    sr = SpinSystem(IsingModel(1), Reciprocal(), [4, 4])
    @test sr.nr isa Reciprocal
end

@testset "Nonreciprocal initialization" begin
    rng = MersenneTwister(7)
    sb = NonreciprocalBlumeCapel([4, 4]; κ=0.3, D=0.2)

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
    si = NonreciprocalIsing([4, 4]; κ=0.0)
    @test_throws ErrorException init!(si, :zero)
end

@testset "κ=0 reduces to equilibrium energy/magnetization" begin
    rng = MersenneTwister(2026)

    # Ising: compare against IsingLattice on an identical random configuration.
    nr = NonreciprocalIsing([6, 6]; κ=0.0, J=1)
    init!(nr, :random, rng=rng)
    iso = Ising([6, 6]; J=1)
    iso.spins .= nr.spins
    MCXSpins._recompute_cached!(iso)
    @test energy(nr) == energy(iso)
    @test magnetization(nr) == magnetization(iso)

    # Blume-Capel: same, including the crystal-field term.
    nb = NonreciprocalBlumeCapel([6, 6]; κ=0.0, D=0.75, J=1)
    init!(nb, :random, rng=rng)
    bc = BlumeCapel([6, 6]; J=1, D=0.75)
    bc.spins .= nb.spins
    MCXSpins._recompute_cached!(bc)
    @test energy(nb) ≈ energy(bc)
    @test magnetization(nb) == magnetization(bc)
end

@testset "Vision-cone asymmetry (field_components)" begin
    sys = NonreciprocalIsing([4, 4]; κ=0.5)
    # Neighbors of site 1 are (4,2,13,5): odd index (4,13) = -dir, even index (2,5) = +dir.
    fill!(sys.spins, Int8(0))
    sys.spins[4] = 1; sys.spins[13] = 1        # -dir sum = +2
    sys.spins[2] = -1; sys.spins[5] = -1       # +dir sum = -2

    @test MCXSpins.field_components(sys, 1, Int8(1))  == (0, -2)   # up spin sees +dir
    @test MCXSpins.field_components(sys, 1, Int8(-1)) == (0, 2)    # down spin sees -dir
    @test MCXSpins.field_components(sys, 1, Int8(0))  == (0, 0)    # no cone

    rec = SpinSystem(IsingModel(1), Reciprocal(), [4, 4])
    rec.spins .= sys.spins
    @test MCXSpins.field_components(rec, 1, Int8(1)) == (0, 0)     # reciprocal: no forward term
end

@testset "spin_flip! sanity and bookkeeping (Glauber and Metropolis)" begin
    for makealg in (rng -> Glauber(rng; β=0.4), rng -> Metropolis(rng; β=0.4))
        rng = MersenneTwister(11)
        sys = NonreciprocalBlumeCapel([8, 8]; κ=0.3, D=0.5)
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

        # cache stays consistent with a full recompute
        @test sys.cache.mag == sum(Int, sys.spins)
        @test sys.cache.spin2 == sum(s -> Int(s)^2, sys.spins)
        @test magnetization(sys) == sys.cache.mag
        @test isfinite(energy(sys))
        @test correlation_length(sys) >= 0.0
    end
end

@testset "Nonreciprocity shifts ordering (Tc grows with κ)" begin
    # Low-temperature smoke test: at fixed β below the reciprocal Tc, larger κ (which raises Tc)
    # should not destroy order. We check that |m| stays high across κ at a cold temperature.
    function mean_abs_m(κ; β, L=16, warmup=200, samples=200, seed=99)
        rng = MersenneTwister(seed)
        sys = NonreciprocalIsing([L, L]; κ=κ)
        init!(sys, :up)
        alg = Glauber(rng; β=β)
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
