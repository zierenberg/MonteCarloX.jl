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

    # d_logweight should be synchronized after simple update
    ens = ensemble(alg)
    for k in 1:length(ens.d_logweight)
        pass &= check(ens.d_logweight[k] ≈ ens.logweight.values[k+1] - ens.logweight.values[k],
                       "d_logweight synchronized after simple update (k=$k)\n")
    end

    # unsupported mode throws
    threw = try; update!(ensemble(alg); mode=:notavail); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "unsupported mode throws\n")

    return pass
end

function test_multicanonical_set_logweight()
    bins = 0.0:1.0:6.0
    alg = Multicanonical(MulticanonicalEnsemble(bins))
    pass = true

    # set! on ensemble synchronizes d_logweight
    set!(ensemble(alg), (1.0, 4.0), x -> 10.0 + x)
    ens = ensemble(alg)
    for k in 1:length(ens.d_logweight)
        pass &= check(ens.d_logweight[k] ≈ ens.logweight.values[k+1] - ens.logweight.values[k],
                       "d_logweight sync after set! on ensemble (k=$k)\n")
    end

    # set! on BinnedObject directly (via logweight accessor) also works
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

function test_extend()
    pass = true

    # --- discrete binning, :low ---
    bins = 0:1:5  # centers at 0,1,2,3,4,5
    ens = MulticanonicalEnsemble(BinnedObject(bins, 0.0); record_visits=false)
    # set! via ensemble to keep d_logweight synchronized
    set!(ens, 0:1:5, x -> x >= 3 ? Float64(x - 2) : 0.0)

    extend!(ens, :low; anchor=3, slope=-0.5)
    # lw(3) = 1.0, so lw(x) = 1.0 + (-0.5)*(x - 3) for x <= 3
    lw = ens.logweight
    pass &= check(lw.values[1] ≈ 1.0 + (-0.5) * (0.0 - 3.0), "discrete :low: bin 0\n")
    pass &= check(lw.values[2] ≈ 1.0 + (-0.5) * (1.0 - 3.0), "discrete :low: bin 1\n")
    pass &= check(lw.values[3] ≈ 1.0 + (-0.5) * (2.0 - 3.0), "discrete :low: bin 2\n")
    pass &= check(lw.values[4] ≈ 1.0 + (-0.5) * (3.0 - 3.0), "discrete :low: anchor bin\n")
    # bins above anchor untouched
    pass &= check(lw.values[5] ≈ 2.0, "discrete :low: upper bin untouched\n")
    pass &= check(lw.values[6] ≈ 3.0, "discrete :low: last bin untouched\n")

    # --- discrete binning, :high ---
    ens2 = MulticanonicalEnsemble(BinnedObject(bins, 0.0); record_visits=false)
    set!(ens2, 0:1:5, x -> x <= 3 ? Float64(x) : 0.0)

    extend!(ens2, :high; anchor=3, slope=0.5)
    # lw(3) = 3.0, so lw(x) = 3.0 + 0.5*(x - 3) for x >= 3
    lw2 = ens2.logweight
    pass &= check(lw2.values[1] ≈ 0.0, "discrete :high: lower bin untouched\n")
    pass &= check(lw2.values[4] ≈ 3.0 + 0.5 * (3.0 - 3.0), "discrete :high: anchor bin\n")
    pass &= check(lw2.values[5] ≈ 3.0 + 0.5 * (4.0 - 3.0), "discrete :high: bin 4\n")
    pass &= check(lw2.values[6] ≈ 3.0 + 0.5 * (5.0 - 3.0), "discrete :high: bin 5\n")

    # --- continuous binning, :low ---
    bins_c = 0.0:1.0:6.0  # 7 edges -> 6 bins, centers at 0.5, 1.5, 2.5, 3.5, 4.5, 5.5
    ens_c = MulticanonicalEnsemble(BinnedObject(bins_c, 0.0); record_visits=false)
    set!(ens_c, 0.0:1.0:6.0, x -> x >= 3.5 ? Float64(x - 2.5) : 0.0)

    extend!(ens_c, :low; anchor=3.5, slope=-0.5)
    # lw(3.5) = 1.0, so lw(x) = 1.0 + (-0.5)*(x - 3.5) for x <= 3.5
    lw_c = ens_c.logweight
    pass &= check(lw_c.values[1] ≈ 1.0 + (-0.5) * (0.5 - 3.5), "continuous :low: bin 0.5\n")
    pass &= check(lw_c.values[2] ≈ 1.0 + (-0.5) * (1.5 - 3.5), "continuous :low: bin 1.5\n")
    pass &= check(lw_c.values[3] ≈ 1.0 + (-0.5) * (2.5 - 3.5), "continuous :low: bin 2.5\n")
    pass &= check(lw_c.values[4] ≈ 1.0 + (-0.5) * (3.5 - 3.5), "continuous :low: anchor bin\n")
    # upper bins untouched
    pass &= check(lw_c.values[5] ≈ 2.0, "continuous :low: upper bin untouched\n")

    # --- limit keyword, :low ---
    ens_lim = MulticanonicalEnsemble(BinnedObject(bins, 0.0); record_visits=false)
    set!(ens_lim, 0:1:5, x -> x >= 3 ? Float64(x - 2) : 0.0)
    extend!(ens_lim, :low; anchor=3, slope=-0.5, limit=1)
    lw_lim = ens_lim.logweight
    # bins at 0 is beyond limit — untouched
    pass &= check(lw_lim.values[1] ≈ 0.0, "limit :low: bin 0 untouched\n")
    # bins at 1,2,3 are within [limit, anchor]
    pass &= check(lw_lim.values[2] ≈ 1.0 + (-0.5) * (1.0 - 3.0), "limit :low: bin 1 ramped\n")
    pass &= check(lw_lim.values[3] ≈ 1.0 + (-0.5) * (2.0 - 3.0), "limit :low: bin 2 ramped\n")
    pass &= check(lw_lim.values[4] ≈ 1.0 + (-0.5) * (3.0 - 3.0), "limit :low: anchor bin\n")
    # bins above anchor untouched
    pass &= check(lw_lim.values[5] ≈ 2.0, "limit :low: upper untouched\n")

    # --- limit keyword, :high ---
    ens_lim2 = MulticanonicalEnsemble(BinnedObject(bins, 0.0); record_visits=false)
    set!(ens_lim2, 0:1:5, x -> x <= 3 ? Float64(x) : 0.0)
    extend!(ens_lim2, :high; anchor=3, slope=0.5, limit=4)
    lw_lim2 = ens_lim2.logweight
    # bins below anchor untouched
    pass &= check(lw_lim2.values[1] ≈ 0.0, "limit :high: lower untouched\n")
    # bins at 3,4 within [anchor, limit]
    pass &= check(lw_lim2.values[4] ≈ 3.0 + 0.5 * (3.0 - 3.0), "limit :high: anchor bin\n")
    pass &= check(lw_lim2.values[5] ≈ 3.0 + 0.5 * (4.0 - 3.0), "limit :high: bin 4 ramped\n")
    # bin at 5 beyond limit — untouched
    pass &= check(lw_lim2.values[6] ≈ 0.0, "limit :high: bin 5 untouched\n")

    # --- invalid direction ---
    threw = try; extend!(ens, :middle; anchor=3, slope=0.0); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "invalid direction throws\n")

    return pass
end

function test_smooth()
    pass = true

    # smooth! modifies d_logweight directly
    bins = 0:1:4  # discrete binning: centers at 0,1,2,3,4; 4 d_logweight entries
    ens = MulticanonicalEnsemble(BinnedObject(bins, 0.0); record_visits=false, warn_overwrite=false)
    # set logweight: [0, 0, 10, 0, 0] → d_logweight: [0, 10, -10, 0]
    set!(ens, 0:1:4, x -> x == 2 ? 10.0 : 0.0)

    smooth!(ens, (0, 4); window=3)
    # d_logweight before: [0, 10, -10, 0]
    # after window=3 smoothing:
    #   k=1: avg of d_lw[1:2] = (0+10)/2 = 5       (clipped at left boundary)
    #   k=2: avg of d_lw[1:3] = (0+10-10)/3 = 0
    #   k=3: avg of d_lw[2:4] = (10-10+0)/3 = 0
    #   k=4: avg of d_lw[3:4] = (-10+0)/2 = -5     (clipped at right boundary)
    pass &= check(ens.d_logweight[1] ≈ 5.0, "smoothed d_logweight[1]\n")
    pass &= check(ens.d_logweight[2] ≈ 0.0, "smoothed d_logweight[2]\n")
    pass &= check(ens.d_logweight[3] ≈ 0.0, "smoothed d_logweight[3]\n")
    pass &= check(ens.d_logweight[4] ≈ -5.0, "smoothed d_logweight[4]\n")
    # logweight reconstructed from smoothed d_logweight
    pass &= check(ens.logweight.values[1] ≈ 0.0, "smoothed logweight[1]\n")
    pass &= check(ens.logweight.values[2] ≈ 5.0, "smoothed logweight[2]\n")
    pass &= check(ens.logweight.values[3] ≈ 5.0, "smoothed logweight[3]\n")
    pass &= check(ens.logweight.values[4] ≈ 5.0, "smoothed logweight[4]\n")
    pass &= check(ens.logweight.values[5] ≈ 0.0, "smoothed logweight[5]\n")

    return pass
end

function test_smooth_window()
    pass = true

    # smooth_window on the ensemble smooths during _integrate! without modifying raw d_logweight
    bins = 0:1:4
    ens = MulticanonicalEnsemble(BinnedObject(bins, 0.0); record_visits=false, smooth_window=3)
    # set raw d_logweight: [0, 10, -10, 0]
    set!(ens, 0:1:4, x -> x == 2 ? 10.0 : 0.0)
    # d_logweight should be [0, 10, -10, 0]
    pass &= check(ens.d_logweight[1] ≈ 0.0, "raw d_logweight[1] preserved\n")
    pass &= check(ens.d_logweight[2] ≈ 10.0, "raw d_logweight[2] preserved\n")
    pass &= check(ens.d_logweight[3] ≈ -10.0, "raw d_logweight[3] preserved\n")
    pass &= check(ens.d_logweight[4] ≈ 0.0, "raw d_logweight[4] preserved\n")

    # set! uses _differentiate!, not _integrate!, so logweight still matches direct values.
    # Trigger _integrate! to see the smooth_window effect.
    MonteCarloX._integrate!(ens)

    # smoothed d_logweight with window=3:
    #   k=1: avg(d_lw[1:2]) = (0+10)/2 = 5.0
    #   k=2: avg(d_lw[1:3]) = (0+10-10)/3 = 0.0
    #   k=3: avg(d_lw[2:4]) = (10-10+0)/3 = 0.0
    #   k=4: avg(d_lw[3:4]) = (-10+0)/2 = -5.0
    # _integrate! uses the running sum:
    #   W[2] = W[1] + smooth(1) = 0 + 5 = 5
    #   W[3] = W[2] + smooth(2) = 5 + 0 = 5
    #   W[4] = W[3] + smooth(3) = 5 + 0 = 5
    #   W[5] = W[4] + smooth(4) = 5 + (-5) = 0
    pass &= check(ens.logweight.values[1] ≈ 0.0, "smooth_window logweight[1]\n")
    pass &= check(ens.logweight.values[2] ≈ 5.0, "smooth_window logweight[2]\n")
    pass &= check(ens.logweight.values[3] ≈ 5.0, "smooth_window logweight[3]\n")
    pass &= check(ens.logweight.values[4] ≈ 5.0, "smooth_window logweight[4]\n")
    pass &= check(ens.logweight.values[5] ≈ 0.0, "smooth_window logweight[5]\n")

    # raw d_logweight is NOT modified by _integrate!
    pass &= check(ens.d_logweight[1] ≈ 0.0, "raw d_logweight[1] still preserved\n")
    pass &= check(ens.d_logweight[2] ≈ 10.0, "raw d_logweight[2] still preserved\n")
    pass &= check(ens.d_logweight[3] ≈ -10.0, "raw d_logweight[3] still preserved\n")
    pass &= check(ens.d_logweight[4] ≈ 0.0, "raw d_logweight[4] still preserved\n")

    return pass
end

function test_recursive_weight_update()
    pass = true

    bins = 0:1:4  # discrete: centers at 0,1,2,3,4
    ens = MulticanonicalEnsemble(bins)

    # first iteration: uniform-ish histogram
    ens.histogram.values .= [10.0, 20.0, 30.0, 20.0, 10.0]
    update!(ens; mode=:recursive)
    pass &= check(all(isfinite.(ens.log_cumweight)), "log_cumweight initialized\n")
    w1 = copy(ens.logweight.values)

    # second iteration: biased histogram (walker stuck at high end)
    ens.histogram.values .= [0.0, 0.0, 5.0, 50.0, 100.0]
    update!(ens; mode=:recursive)
    w2 = copy(ens.logweight.values)

    # the recursive update should preserve shape in unvisited region better
    # than simple update would.
    pass &= check(all(isfinite.(w2)), "all weights finite after recursive update\n")

    # third iteration: now bias the other way
    ens.histogram.values .= [100.0, 50.0, 5.0, 0.0, 0.0]
    update!(ens; mode=:recursive)
    w3 = copy(ens.logweight.values)
    pass &= check(all(isfinite.(w3)), "all weights finite after third recursive update\n")

    # verify accumulated precision grows across iterations
    # log_cumweight[1] covers the transition between bins 0-1, sampled in
    # iterations 1 and 3 — should have accumulated more than after iter 1
    pass &= check(ens.log_cumweight[1] > -Inf, "accumulated precision is finite for sampled transitions\n")

    return pass
end

function test_recursive_weight_correction()
    pass = true

    # verify that the recursive update accounts for sampling weights
    bins = 0:1:2  # 3 bins, 2 transitions
    ens = MulticanonicalEnsemble(bins; warn_overwrite=false)

    # set non-trivial initial weights
    set!(ens, 0:1:2, x -> Float64(x))
    # logweight = [0, 1, 2], d_logweight = [1, 1]

    # flat histogram under these weights means density of states is NOT flat
    ens.histogram.values .= [100.0, 100.0, 100.0]
    update!(ens; mode=:recursive)

    # with flat histogram and initial d_lw = [1, 1], the measured d_lw_ideal is:
    # d_lw_actual + log(H[k]) - log(H[k+1]) = 1 + 0 = 1 for both transitions
    # so d_lw should not change (the weights are already consistent with flat histogram)
    pass &= check(ens.d_logweight[1] ≈ 1.0, "flat histogram preserves d_logweight[1]\n")
    pass &= check(ens.d_logweight[2] ≈ 1.0, "flat histogram preserves d_logweight[2]\n")

    return pass
end

function test_visited_range()
    pass = true

    bins = 0:1:10
    rng = MersenneTwister(42)
    alg = Multicanonical(rng, bins)
    ens = ensemble(alg)

    # initially empty
    vmin, vmax = visited_range(ens)
    pass &= check(vmin == Inf && vmax == -Inf, "visited_range empty before any visits\n")

    # simulate visits via accept!
    accept!(alg, 3.0, 3.0)
    accept!(alg, 7.0, 3.0)
    vmin, vmax = visited_range(ens)
    pass &= check(vmin ≈ 3.0, "visited_min after visits\n")
    pass &= check(vmax ≈ 7.0, "visited_max after visits\n")

    # reset! does NOT shrink the visited range
    reset!(alg)
    vmin, vmax = visited_range(ens)
    pass &= check(vmin ≈ 3.0, "visited_min preserved after reset!\n")
    pass &= check(vmax ≈ 7.0, "visited_max preserved after reset!\n")

    # new visit only expands range
    accept!(alg, 5.0, 5.0)
    vmin, vmax = visited_range(ens)
    pass &= check(vmin ≈ 3.0, "visited_min unchanged by interior visit\n")
    pass &= check(vmax ≈ 7.0, "visited_max unchanged by interior visit\n")

    accept!(alg, 1.0, 5.0)
    vmin, vmax = visited_range(ens)
    pass &= check(vmin ≈ 1.0, "visited_min expanded\n")
    pass &= check(vmax ≈ 7.0, "visited_max unchanged\n")

    return pass
end

function test_warn_overwrite_toggle()
    pass = true

    bins = 0:1:4
    # warn_overwrite=true (default) — tested implicitly elsewhere
    ens = MulticanonicalEnsemble(bins; warn_overwrite=false)
    pass &= check(ens.warn_overwrite == false, "warn_overwrite can be set to false\n")

    # accumulate some precision
    ens.histogram.values .= [10.0, 20.0, 30.0, 20.0, 10.0]
    update!(ens; mode=:recursive)
    pass &= check(ens.log_cumweight[1] > -Inf, "precision accumulated\n")

    # extend! should NOT warn when warn_overwrite=false
    # (just verify no error is thrown)
    extend!(ens, :low; anchor=3, slope=-0.5)
    pass &= check(true, "extend! with warn_overwrite=false does not error\n")

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
    @testset "recursive weight correction" begin
        @test test_recursive_weight_correction()
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
    @testset "extend" begin
        @test test_extend()
    end
@testset "smooth" begin
        @test test_smooth()
    end
    @testset "smooth_window" begin
        @test test_smooth_window()
    end
    @testset "visited_range" begin
        @test test_visited_range()
    end
    @testset "warn_overwrite toggle" begin
        @test test_warn_overwrite_toggle()
    end
end
