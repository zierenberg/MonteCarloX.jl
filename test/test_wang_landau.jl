using MonteCarloX
using Random
using StatsBase
using Test

function test_wang_landau_local_update(; verbose=false)
    rng = MersenneTwister(780)
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedLogWeight(bins, 0.0)
    wl = WangLandau(rng, lw; logf=log(2.0))

    x = 1.2
    w0 = lw[x]
    pass &= update_weight!(wl, x) === nothing

    pass &= lw[x] == w0 - wl.logf

    if verbose
        println("Wang-Landau local update:")
        println("  weight before: $(w0), after: $(lw[x])")
    end

    return pass
end

function test_wang_landau_update_f(; verbose=false)
    rng = MersenneTwister(781)

    bins = 0.0:1.0:4.0
    lw = BinnedLogWeight(bins, 0.0)
    wl = WangLandau(rng, lw; logf=1.0)

    logf0 = wl.logf
    pass = update_f!(wl) === nothing && wl.logf == 0.5 * logf0

    if verbose
        println("Wang-Landau update_f!: logf0=$(logf0), logf1=$(wl.logf)")
    end

    return pass
end

function test_wang_landau_default_rng(; verbose=false)
    bins = 0.0:1.0:4.0
    lw = BinnedLogWeight(bins, 0.0)
    wl = WangLandau(lw; logf=log(2.0))

    pass = wl.rng === Random.GLOBAL_RNG

    if verbose
        println("Wang-Landau default RNG: $(pass)")
    end

    return pass
end

function run_wang_landau_testsets(; verbose=false)
    @testset "Wang-Landau" begin
        @testset "Local update" begin
            @test test_wang_landau_local_update(verbose=verbose)
        end
        @testset "Update f" begin
            @test test_wang_landau_update_f(verbose=verbose)
        end
        @testset "Default RNG" begin
            @test test_wang_landau_default_rng(verbose=verbose)
        end
    end
    return true
end
