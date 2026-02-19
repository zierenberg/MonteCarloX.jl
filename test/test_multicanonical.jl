using MonteCarloX
using Random
using StatsBase
using Test

function test_multicanonical_weight_update_inplace(; verbose=false)
    rng = MersenneTwister(901)
    pass = true

    bins = 0.0:1.0:4.0
    lw_hist = Histogram((collect(bins),), zeros(Float64, length(bins) - 1))
    lw = TabulatedLogWeight(lw_hist)
    alg = Multicanonical(rng, lw)

    hist = fit(Histogram, [0.2, 0.8, 1.1, 2.5, 2.7], bins)

    w_before = copy(lw.table.weights)
    pass &= update_weights!(alg, hist; mode=:simple) === nothing

    expected = copy(w_before)
    for i in eachindex(expected)
        h = hist.weights[i]
        if h > 0
            expected[i] -= log(h)
        end
    end

    pass &= all(isapprox.(lw.table.weights, expected))

    if verbose
        println("Multicanonical in-place update:")
        println("  before: $(w_before)")
        println("  after:  $(lw.table.weights)")
    end

    return pass
end

function test_multicanonical_bin_compatibility(; verbose=false)
    rng = MersenneTwister(902)
    bins_lw = 0.0:1.0:4.0
    bins_h = 0.0:0.5:4.0

    lw_hist = Histogram((collect(bins_lw),), zeros(Float64, length(bins_lw) - 1))
    lw = TabulatedLogWeight(lw_hist)
    alg = Multicanonical(rng, lw)

    hist_bad = fit(Histogram, [0.25, 1.25, 2.75], bins_h)

    pass = true
    pass &= try
        update_weights!(alg, hist_bad; mode=:simple)
        false
    catch err
        err isa ArgumentError
    end

    if verbose
        println("Multicanonical bin compatibility: $(pass)")
    end

    return pass
end

function run_multicanonical_testsets(; verbose=false)
    @testset "Multicanonical" begin
        @testset "In-place update" begin
            @test test_multicanonical_weight_update_inplace(verbose=verbose)
        end
        @testset "Bin compatibility" begin
            @test test_multicanonical_bin_compatibility(verbose=verbose)
        end
    end
    return true
end
