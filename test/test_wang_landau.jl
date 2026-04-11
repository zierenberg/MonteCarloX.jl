using MonteCarloX
using Random
using StatsBase
using Test

function test_wang_landau_accept()
    rng = MersenneTwister(42)
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    alg = WangLandau(rng, lw)

    step = 0.1
    function update!(x::Float64, alg)::Float64
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
    pass &= check(acceptance_rate(alg) >= 0.0 && acceptance_rate(alg) <= 1.0, "acceptance rate in [0, 1]\n")

    reset!(alg)
    pass &= check(alg.accepted == 0, "accepted reset\n")

    return pass
end

function test_wang_landau_local_update()
    rng = MersenneTwister(780)
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    wl = WangLandau(rng, lw; logf=log(2.0))

    x = 1.2
    w0 = lw[x]
    pass &= check(accept!(wl, x, x) == true, "self-move accepted\n")
    pass &= check(lw[x] == w0 - ensemble(wl).logf, "weight updated by logf\n")

    return pass
end

function test_wang_landau_update_f()
    rng = MersenneTwister(781)

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    wl = WangLandau(rng, lw; logf=1.0)

    logf0 = ensemble(wl).logf
    pass = true
    pass &= check(update!(ensemble(wl)) === nothing, "update! returns nothing\n")
    pass &= check(ensemble(wl).logf == 0.5 * logf0, "logf halved\n")

    return pass
end

function test_wang_landau_default_rng()
    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    wl = WangLandau(lw; logf=log(2.0))

    pass = check(wl.rng === Random.GLOBAL_RNG, "default RNG is GLOBAL_RNG\n")

    return pass
end

@testset "Wang-Landau" begin
    @testset "Accept/reject" begin
        @test test_wang_landau_accept()
    end
    @testset "Local update" begin
        @test test_wang_landau_local_update()
    end
    @testset "Update f" begin
        @test test_wang_landau_update_f()
    end
    @testset "Default RNG" begin
        @test test_wang_landau_default_rng()
    end
end
