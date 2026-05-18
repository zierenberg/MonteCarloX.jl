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

function test_roundtrips()
    pass = true

    # constructor validates x_min < x_max
    threw = try; Roundtrips(5.0, 2.0); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "Roundtrips constructor throws on x_min >= x_max\n")

    rt = Roundtrips(0.0, 10.0)
    pass &= check(rt.count == 0, "initial count is 0\n")
    pass &= check(rt.at_boundary == 0, "initial boundary state is 0\n")

    # reach min boundary
    update!(rt, -1.0)
    pass &= check(rt.at_boundary == -1, "at_boundary after reaching min\n")
    pass &= check(rt.count == 0, "no roundtrip yet\n")

    # stay in middle — no change
    update!(rt, 5.0)
    pass &= check(rt.at_boundary == -1, "boundary unchanged in middle\n")

    # reach max boundary — completes one round trip
    update!(rt, 11.0)
    pass &= check(rt.at_boundary == 1, "at_boundary after reaching max\n")
    pass &= check(rt.count == 1, "one roundtrip completed\n")

    # reach min again — second round trip
    update!(rt, -0.5)
    pass &= check(rt.count == 2, "second roundtrip completed\n")

    # starting from max without prior min — no roundtrip
    rt2 = Roundtrips(0.0, 10.0)
    update!(rt2, 11.0)
    pass &= check(rt2.count == 0, "no roundtrip from cold start at max\n")
    update!(rt2, -1.0)
    pass &= check(rt2.count == 1, "roundtrip after max->min\n")

    # reset
    reset!(rt)
    pass &= check(rt.count == 0, "count reset\n")
    pass &= check(rt.at_boundary == 0, "boundary state reset\n")

    return pass
end

function test_flatness()
    pass = true

    # perfectly flat histogram
    bins = 0.0:1.0:4.0
    h = BinnedObject(bins, 0.0)
    h.values .= [10.0, 10.0, 10.0, 10.0]
    pass &= check(flatness(h, 0.0, 4.0) ≈ 1.0, "flat histogram has flatness 1.0\n")
    pass &= check(flatness(h, 0.0, 4.0; criterion=:mean_over_min) ≈ 1.0, "flat histogram mean_over_min 1.0\n")

    # non-flat histogram
    h.values .= [1.0, 2.0, 3.0, 4.0]
    f = flatness(h, 0.0, 4.0; criterion=:max_over_mean)
    pass &= check(f ≈ 4.0 / 2.5, "max_over_mean correct\n")
    f2 = flatness(h, 0.0, 4.0; criterion=:mean_over_min)
    pass &= check(f2 ≈ 2.5 / 1.0, "mean_over_min correct\n")

    # with zeros — only occupied bins count
    h.values .= [0.0, 5.0, 5.0, 0.0]
    pass &= check(flatness(h, 0.0, 4.0) ≈ 1.0, "zeros excluded, flat occupied bins\n")

    # all zeros
    h.values .= [0.0, 0.0, 0.0, 0.0]
    pass &= check(flatness(h, 0.0, 4.0) == Inf, "all zeros returns Inf\n")

    # sub-range
    h.values .= [1.0, 10.0, 10.0, 100.0]
    f_sub = flatness(h, 1.0, 3.0; criterion=:max_over_mean)
    pass &= check(f_sub ≈ 1.0, "sub-range flatness\n")

    # unsupported criterion
    threw = try; flatness(h, 0.0, 4.0; criterion=:bad); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "unsupported criterion throws\n")

    return pass
end

function test_extrapolate()
    pass = true

    # --- discrete binning ---
    bins = 0:1:5  # centers at 0,1,2,3,4,5
    lw = BinnedObject(bins, 0.0)
    lw.values .= [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]

    extrapolate!(lw, (0, 2); anchor=3, slope=-0.5)
    # lw(3) = 1.0, so lw(x) = 1.0 + (-0.5)*(x - 3)
    pass &= check(lw.values[1] ≈ 1.0 + (-0.5) * (0.0 - 3.0), "discrete: extrapolate bin 0\n")
    pass &= check(lw.values[2] ≈ 1.0 + (-0.5) * (1.0 - 3.0), "discrete: extrapolate bin 1\n")
    pass &= check(lw.values[3] ≈ 1.0 + (-0.5) * (2.0 - 3.0), "discrete: extrapolate bin 2\n")
    # bins outside range untouched
    pass &= check(lw.values[4] ≈ 1.0, "discrete: anchor bin untouched\n")
    pass &= check(lw.values[6] ≈ 3.0, "discrete: upper bins untouched\n")

    # --- continuous binning ---
    bins_c = 0.0:1.0:6.0  # 7 edges -> 6 bins, centers at 0.5, 1.5, 2.5, 3.5, 4.5, 5.5
    lw_c = BinnedObject(bins_c, 0.0)
    lw_c.values .= [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]

    extrapolate!(lw_c, (0.0, 2.0); anchor=3.5, slope=-0.5)
    # lw(3.5) = 1.0, so lw(x) = 1.0 + (-0.5)*(x - 3.5) for centers in [0.0, 2.0]
    pass &= check(lw_c.values[1] ≈ 1.0 + (-0.5) * (0.5 - 3.5), "continuous: extrapolate bin 0.5\n")
    pass &= check(lw_c.values[2] ≈ 1.0 + (-0.5) * (1.5 - 3.5), "continuous: extrapolate bin 1.5\n")
    # bin at center 2.5 is outside range [0.0, 2.0]
    pass &= check(lw_c.values[3] ≈ 0.0, "continuous: bin outside range untouched\n")
    # upper bins untouched
    pass &= check(lw_c.values[4] ≈ 1.0, "continuous: anchor bin untouched\n")

    return pass
end

function test_interpolate_gaps()
    pass = true

    # --- discrete binning ---
    bins = 0:1:6  # centers at 0,1,...,6
    lw = BinnedObject(bins, 0.0)
    h  = BinnedObject(bins, 0.0)

    # occupied at bins 0,1 and 5,6 with gap at 2,3,4
    lw.values .= [0.0, 1.0, 0.0, 0.0, 0.0, 5.0, 6.0]
    h.values  .= [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    interpolate_gaps!(lw, h, (0, 6))

    # gap indices 3-5 interpolated between lw[2]=1.0 and lw[6]=5.0
    pass &= check(lw.values[3] ≈ 2.0, "discrete: gap interpolation bin 2\n")
    pass &= check(lw.values[4] ≈ 3.0, "discrete: gap interpolation bin 3\n")
    pass &= check(lw.values[5] ≈ 4.0, "discrete: gap interpolation bin 4\n")
    pass &= check(lw.values[2] ≈ 1.0, "discrete: occupied bin unchanged\n")
    pass &= check(lw.values[6] ≈ 5.0, "discrete: occupied bin unchanged\n")

    # --- continuous binning ---
    bins_c = 0.0:1.0:5.0  # 6 edges -> 5 bins, centers at 0.5, 1.5, 2.5, 3.5, 4.5
    lw_c = BinnedObject(bins_c, 0.0)
    h_c  = BinnedObject(bins_c, 0.0)

    lw_c.values .= [1.0, 0.0, 0.0, 4.0, 5.0]
    h_c.values  .= [1.0, 0.0, 0.0, 1.0, 1.0]

    interpolate_gaps!(lw_c, h_c, (0.0, 5.0))

    # gap at indices 2,3 interpolated between lw_c[1]=1.0 and lw_c[4]=4.0
    pass &= check(lw_c.values[2] ≈ 2.0, "continuous: gap interpolation\n")
    pass &= check(lw_c.values[3] ≈ 3.0, "continuous: gap interpolation\n")
    pass &= check(lw_c.values[1] ≈ 1.0, "continuous: occupied unchanged\n")
    pass &= check(lw_c.values[4] ≈ 4.0, "continuous: occupied unchanged\n")

    return pass
end

function test_smooth()
    pass = true

    bins = 0:1:4  # discrete binning: centers at 0,1,2,3,4
    lw = BinnedObject(bins, 0.0)
    lw.values .= [0.0, 0.0, 10.0, 0.0, 0.0]

    smooth!(lw, (0, 4); window=3)
    # bin at index 3 (value 2.0): avg of [0, 10, 0] = 10/3
    pass &= check(lw.values[3] ≈ 10.0 / 3.0, "smoothed center bin\n")
    # bin at index 2 (value 1.0): avg of [0, 0, 10] = 10/3
    pass &= check(lw.values[2] ≈ 10.0 / 3.0, "smoothed left neighbor\n")
    # edge bin at index 1: avg of [0, 0] = 0 (window clipped)
    pass &= check(lw.values[1] ≈ 0.0, "edge bin smoothed\n")

    return pass
end

function test_recursive_weight_update()
    pass = true

    bins = 0:1:4  # discrete: centers at 0,1,2,3,4
    ens = MulticanonicalEnsemble(bins)

    # first iteration: uniform-ish histogram
    ens.histogram.values .= [10.0, 20.0, 30.0, 20.0, 10.0]
    update!(ens; mode=:recursive)
    pass &= check(ens.log_p_acc !== nothing, "accumulator initialized\n")
    w1 = copy(ens.logweight.values)

    # second iteration: biased histogram (walker stuck at high end)
    ens.histogram.values .= [0.0, 0.0, 5.0, 50.0, 100.0]
    update!(ens; mode=:recursive)
    w2 = copy(ens.logweight.values)

    # the recursive update should preserve shape in unvisited region better
    # than simple update would. Check that accumulated precision prevents
    # complete destruction of weights at low end.
    # With simple update, bins 0,1 would be unchanged (h=0 -> no update).
    # With recursive, the entropy is reconstructed from high to low using
    # accumulated statistics, so bins 0,1 still get reasonable values.
    pass &= check(all(isfinite.(w2)), "all weights finite after recursive update\n")

    # third iteration: now bias the other way
    ens.histogram.values .= [100.0, 50.0, 5.0, 0.0, 0.0]
    update!(ens; mode=:recursive)
    w3 = copy(ens.logweight.values)
    pass &= check(all(isfinite.(w3)), "all weights finite after third recursive update\n")

    # verify accumulated precision grows across iterations
    # log_p_acc[2] covers the transition between bins 1-2, sampled in
    # iterations 1 and 3 — should have accumulated more than after iter 1
    pass &= check(ens.log_p_acc[2] > -Inf, "accumulated precision is finite for sampled transitions\n")

    return pass
end

@testset "Multicanonical" begin
    @testset "accept and reset" begin
        @test test_multicanonical_accept_and_reset()
    end
    @testset "weight update" begin
        @test test_multicanonical_weight_update()
    end
    @testset "recursive weight update" begin
        @test test_recursive_weight_update()
    end
    @testset "set logweight" begin
        @test test_multicanonical_set_logweight()
    end
    @testset "roundtrips" begin
        @test test_roundtrips()
    end
    @testset "flatness" begin
        @test test_flatness()
    end
    @testset "extrapolate" begin
        @test test_extrapolate()
    end
    @testset "interpolate_gaps" begin
        @test test_interpolate_gaps()
    end
    @testset "smooth" begin
        @test test_smooth()
    end
end
