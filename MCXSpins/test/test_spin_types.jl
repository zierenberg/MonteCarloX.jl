# Spin types: state sets across S, proposal correctness, allocation-freeness, and the
# update-owned XY proposal width.

@testset "Spin(S) state sets" begin
    @test states(Spin(1//2)) == (Int8(-1), Int8(1))
    @test states(Spin(1)) == (Int8(-1), Int8(0), Int8(1))
    @test states(Spin(3//2)) == (Int8(-3), Int8(-1), Int8(1), Int8(3))
    @test states(Spin(2)) == (Int8(-2), Int8(-1), Int8(0), Int8(1), Int8(2))

    # the m = −S…S ladder closes only for (half-)integer S — anything else is refused
    @test_throws ArgumentError Spin(1//3)
    @test_throws ArgumentError Spin(0.7)
    @test_throws ArgumentError Spin(0)
    @test_throws ArgumentError Spin(-1)
end

@testset "Discrete proposals: skip-current, full coverage, allocation-free" begin
    rng = MersenneTwister(17)
    for S in (1//2, 1, 3//2)
        spintype = Spin(S)
        for s_old in states(spintype)
            draws = [propose_state(rng, spintype, s_old) for _ in 1:300]
            @test all(!=(s_old), draws)
            @test Set(draws) == Set(s for s in states(spintype) if s != s_old)
        end
    end

    # states(spintype) is a compile-time constant: no tuple is built at runtime
    alloc_probe(r, st, s) = @allocated propose_state(r, st, s)
    st = Spin(3//2)
    alloc_probe(rng, st, Int8(1))
    @test alloc_probe(rng, st, Int8(1)) == 0
end

@testset "Continuous spin types" begin
    rng = MersenneTwister(23)
    s_xy = propose_state(rng, XYSpin(), cis(0.3), 0.5)
    @test abs(abs(s_xy) - 1) < 1e-12                       # unit modulus preserved
    @test abs(angle(s_xy) - 0.3) <= 0.5 + 1e-12            # within the proposal half-width

    # uniform on the unit sphere
    @test all(abs(sum(abs2, MCXSpins.random_state(rng, HeisenbergSpin())) - 1) < 1e-12
              for _ in 1:100)
end

@testset "XY proposal width lives with the update call" begin
    sys = XYSystem([4, 4]; J=1.0)
    init!(sys, :random, rng=MersenneTwister(2))
    alg = MetropolisAlgorithm(MersenneTwister(3); β=0.7)
    before = copy(sys.spins)
    for _ in 1:2000
        spin_flip!(sys, alg; Δθ=0.5)
    end
    @test sys.spins != before
    @test all(s -> abs(abs(s) - 1) < 1e-9, sys.spins)
    @test 0 < acceptance_rate(alg) < 1
    @test_throws UndefKeywordError spin_flip!(sys, alg)   # no Δθ keyword: XY proposal undefined
end
