using MonteCarloX
using Random
using StatsBase
using Test


function test_wang_landau_accept(; verbose=false)
    rng = MersenneTwister(42)
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedLogWeight(bins, 0.0)
    alg = WangLandau(rng, lw)

    step = 0.1
    function update!(x::Float64, alg::WangLandau)::Float64
        x_new = x + randn(alg.rng) * step
        if accept!(alg, x_new, x)
            return x_new
        else
            return x
        end
    end

    # updates
    x = 2.0  # start in bin 2
    for _ in 1:10
        x = update!(x, alg)
    end
    
    # test num_accepts
    pass &= alg.accepted >= 0 && alg.accepted <= 10
    # test acceptance rate (between 0 and 1 since log_weight is zero, so acceptance should be 1)
    pass &= acceptance_rate(alg) >= 0.0 && acceptance_rate(alg) <= 1.0

    # reset
    reset!(alg)
    pass &= alg.accepted == 0
    
    return pass
end

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
        @testset "Accept/reject" begin
            @test test_wang_landau_accept(verbose=verbose)
        end
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
