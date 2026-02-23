using MonteCarloX
using Random
using StatsBase
using Test

function test_multicanonical_weight_update_inplace(; verbose=false)
    rng = MersenneTwister(901)
    pass = true

    bins = 0.0:1.0:4.0
    lw = TabulatedLogWeight(bins, 0.0)
    alg = Multicanonical(rng, lw)

    w_before = copy(lw.histogram.weights)
    alg.histogram.weights = [0.2, 0.8, 1.1, 2.5]

    if verbose
        # print the indices of the bins that are being updated
        println("logweight edges:", alg.logweight.histogram.edges)
        println("logweight weights:", alg.logweight.histogram.weights)
        println("histogram edges:", alg.histogram.edges)
        println("histogram weights:", alg.histogram.weights)
    end

    pass &= update_weight!(alg) === nothing

    expected = copy(w_before)
    for i in eachindex(expected)
        h = alg.histogram.weights[i]
        if h > 0
            expected[i] -= log(h)
        end
    end

    pass &= all(isapprox.(lw.histogram.weights, expected))

    if verbose
        println("Multicanonical in-place update:")
        println("  before: $(w_before)")
        println("  after:  $(lw.histogram.weights)")
    end

    return pass
end

function test_multicanonical_default_rng(; verbose=false)
    bins = 0.0:1.0:4.0
    lw = TabulatedLogWeight(Histogram((collect(bins),), zeros(Float64, length(bins) - 1)))
    alg = Multicanonical(lw)

    pass = alg.rng === Random.GLOBAL_RNG

    if verbose
        println("Multicanonical default RNG: $(pass)")
    end

    return pass
end

function test_multicanonical_mode(; verbose=false)
    rng = MersenneTwister(902)
    bins_lw = 0.0:1.0:4.0

    lw = TabulatedLogWeight(bins_lw, 0.0)
    alg = Multicanonical(rng, lw)

    pass = true    
    pass &= try
        update_weight!(alg; mode=:notavail)  # unsupported mode should throw
        false
    catch err
        err isa ArgumentError
    end

    if verbose
        println("Multicanonical mode compatibility: $(pass)")
    end

    return pass
end

function run_multicanonical_testsets(; verbose=false)
    @testset "Multicanonical" begin
        @testset "In-place update" begin
            @test test_multicanonical_weight_update_inplace(verbose=verbose)
        end
        @testset "Mode compatibility" begin
            @test test_multicanonical_mode(verbose=verbose)
        end
        @testset "Default RNG" begin
            @test test_multicanonical_default_rng(verbose=verbose)
        end
    end
    return true
end
