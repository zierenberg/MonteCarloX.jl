using MonteCarloX
using Random
using StatsBase
using Test

function test_multicanonical_accept()
    rng = MersenneTwister(42)
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
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

    pass &= check(alg.accepted >= 0 && alg.accepted <= 10, "accepted in [0, 10]\n")
    # log_weight is zero everywhere, so acceptance should be 1
    pass &= check(acceptance_rate(alg) == 1.0, "acceptance rate == 1.0 (flat weights)\n")

    reset!(alg)
    pass &= check(alg.accepted == 0, "accepted reset\n")
    pass &= check(all(iszero, ensemble(alg).histogram.values), "histogram reset\n")

    return pass
end

function test_multicanonical_weight_update_inplace()
    rng = MersenneTwister(901)
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    alg = Multicanonical(rng, lw)

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

    return pass
end

function test_multicanonical_mode()
    rng = MersenneTwister(902)
    bins_lw = 0.0:1.0:4.0

    lw = BinnedObject(bins_lw, 0.0)
    alg = Multicanonical(rng, lw)

    pass = true
    threw = try; update!(ensemble(alg); mode=:notavail); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "unsupported mode throws\n")

    return pass
end

function test_multicanonical_default_rng()
    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    alg = Multicanonical(lw)

    pass = check(alg.rng === Random.GLOBAL_RNG, "default RNG is GLOBAL_RNG\n")

    return pass
end

function test_multicanonical_accept_out_of_bounds()
    rng = MersenneTwister(905)
    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    alg = Multicanonical(rng, lw)

    x_old = 2.0
    x_new = 10.0  # outside right edge

    steps_before = alg.steps
    accepted_before = alg.accepted
    hist_before = copy(ensemble(alg).histogram.values)

    pass = true
    threw = try; accept!(alg, x_new, x_old); false; catch err; err isa BoundsError; end
    pass &= check(threw, "out-of-bounds throws BoundsError\n")
    pass &= check(alg.steps == steps_before, "steps unchanged\n")
    pass &= check(alg.accepted == accepted_before, "accepted unchanged\n")
    pass &= check(all(ensemble(alg).histogram.values .== hist_before), "histogram unchanged\n")

    return pass
end

function test_multicanonical_set_logweight_range_function()
    bins = 0.0:1.0:6.0
    ens = MulticanonicalEnsemble(bins)
    alg = Multicanonical(ens)

    pass = true
    pass &= check(alg.rng === Random.GLOBAL_RNG, "default RNG is GLOBAL_RNG\n")

    fill!(ensemble(alg).logweight.values, 0.0)

    pass &= check(set!(logweight(alg), (1.0, 4.0), x -> 10.0 + x) === nothing, "set! returns nothing\n")
    expected = [0.0, 11.5, 12.5, 13.5, 0.0, 0.0]
    pass &= check(all(isapprox.(ensemble(alg).logweight.values, expected)), "set! restricted range\n")

    pass &= check(set!(logweight(alg), 0.0:1.0:6.0, x -> -x^2) === nothing, "set! full range returns nothing\n")
    centers = get_centers(ensemble(alg).histogram)
    pass &= check(all(isapprox.(ensemble(alg).logweight.values, -centers.^2)), "set! full range values\n")

    # set logweight so that move is not accepted
    x = 0.0
    set!(logweight(alg), (1.0, 6.0), w -> -100.0)
    pass &= check(accept!(alg, 2.0, x) == false, "suppressed acceptance\n")

    return pass
end

function test_multicanonical_set_logweight_range_errors()
    rng = MersenneTwister(904)
    bins = 0.0:1.0:5.0
    lw = BinnedObject(bins, 0.0)
    alg = Multicanonical(rng, lw)

    pass = true
    threw = try; set!(logweight(alg), (100.0, 200.0), x -> x); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "out-of-range set! throws\n")

    return pass
end

@testset "Multicanonical" begin
    @testset "Accept/reject" begin
        @test test_multicanonical_accept()
    end
    @testset "Accept out-of-bounds" begin
        @test test_multicanonical_accept_out_of_bounds()
    end
    @testset "In-place update" begin
        @test test_multicanonical_weight_update_inplace()
    end
    @testset "Mode compatibility" begin
        @test test_multicanonical_mode()
    end
    @testset "Default RNG" begin
        @test test_multicanonical_default_rng()
    end
    @testset "Set logweight range/function" begin
        @test test_multicanonical_set_logweight_range_function()
    end
    @testset "Set logweight range errors" begin
        @test test_multicanonical_set_logweight_range_errors()
    end
end
