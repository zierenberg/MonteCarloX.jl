using MonteCarloX
using Random
using StatsBase
using Test

function test_multicanonical_accept_and_reset()
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)

    # default RNG constructor
    alg_default = Multicanonical(lw)
    pass &= check(alg_default.rng === Random.GLOBAL_RNG, "default RNG is GLOBAL_RNG\n")

    # flat weights: acceptance rate should be 1.0
    rng = MersenneTwister(42)
    alg = Multicanonical(rng, lw)

    step = 0.1
    function update!(x::Float64, alg::AbstractImportanceSampling)::Float64
        x_new = x + randn(alg.rng) * step
        if accept!(alg, x_new, x)
            return x_new
        else
            return x
        end
    end

    x = 2.0
    for _ in 1:10
        x = update!(x, alg)
    end
    pass &= check(acceptance_rate(alg) == 1.0, "acceptance rate == 1.0 (flat weights)\n")

    # accept! records visits in histogram
    pass &= check(sum(ensemble(alg).histogram.values) == alg.steps, "histogram total == steps\n")

    # out-of-bounds proposal throws BoundsError, state unchanged
    steps_before = alg.steps
    threw = try; accept!(alg, 10.0, 2.0); false; catch err; err isa BoundsError; end
    pass &= check(threw, "out-of-bounds throws BoundsError\n")
    pass &= check(alg.steps == steps_before, "steps unchanged after error\n")

    # reset clears counters and histogram
    reset!(alg)
    pass &= check(alg.accepted == 0, "accepted reset\n")
    pass &= check(all(iszero, ensemble(alg).histogram.values), "histogram reset\n")

    return pass
end

function test_multicanonical_weight_update()
    rng = MersenneTwister(901)
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    alg = Multicanonical(rng, lw)

    # in-place weight update from histogram
    w_before = copy(ensemble(alg).logweight.values)
    ensemble(alg).histogram.values .= [0.2, 0.8, 1.1, 2.5]
    pass &= check(update!(ensemble(alg)) === nothing, "update! returns nothing\n")

    expected = copy(w_before)
    for i in eachindex(expected)
        h = ensemble(alg).histogram.values[i]
        if h > 0
            expected[i] -= log(h)
        end
    end
    pass &= check(all(isapprox.(ensemble(alg).logweight.values, expected)), "logweight updated correctly\n")

    # unsupported mode throws
    threw = try; update!(ensemble(alg); mode=:notavail); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "unsupported mode throws\n")

    return pass
end

function test_multicanonical_set_logweight()
    bins = 0.0:1.0:6.0
    alg = Multicanonical(MulticanonicalEnsemble(bins))
    pass = true

    # set! on restricted range
    fill!(ensemble(alg).logweight.values, 0.0)
    pass &= check(set!(logweight(alg), (1.0, 4.0), x -> 10.0 + x) === nothing, "set! returns nothing\n")
    expected = [0.0, 11.5, 12.5, 13.5, 0.0, 0.0]
    pass &= check(all(isapprox.(ensemble(alg).logweight.values, expected)), "set! restricted range\n")

    # set! on full range
    set!(logweight(alg), 0.0:1.0:6.0, x -> -x^2)
    centers = get_centers(ensemble(alg).histogram)
    pass &= check(all(isapprox.(ensemble(alg).logweight.values, -centers.^2)), "set! full range values\n")

    # out-of-range set! throws
    threw = try; set!(logweight(alg), (100.0, 200.0), x -> x); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "out-of-range set! throws\n")

    # suppressed acceptance via extreme weights
    set!(logweight(alg), (1.0, 6.0), w -> -100.0)
    pass &= check(accept!(alg, 2.0, 0.0) == false, "suppressed acceptance\n")

    return pass
end

@testset "Multicanonical" begin
    @testset "accept and reset" begin
        @test test_multicanonical_accept_and_reset()
    end
    @testset "weight update" begin
        @test test_multicanonical_weight_update()
    end
    @testset "set logweight" begin
        @test test_multicanonical_set_logweight()
    end
end
