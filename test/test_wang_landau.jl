using MonteCarloX
using Random
using StatsBase
using Test

function test_wang_landau_accept_and_reset()
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)

    # default RNG constructor
    wl_default = WangLandau(lw; logf=log(2.0))
    pass &= check(wl_default.rng === Random.GLOBAL_RNG, "default RNG is GLOBAL_RNG\n")

    # accept/reject loop with flat weights
    rng = MersenneTwister(42)
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

function test_wang_landau_update_mechanics()
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    wl = WangLandau(MersenneTwister(780), lw; logf=log(2.0))

    # accept! decrements logweight by logf
    x = 1.2
    w0 = lw[x]
    pass &= check(accept!(wl, x, x) == true, "self-move accepted\n")
    pass &= check(lw[x] == w0 - ensemble(wl).logf, "weight updated by logf\n")

    # update! halves logf
    logf0 = ensemble(wl).logf
    pass &= check(update!(ensemble(wl)) === nothing, "update! returns nothing\n")
    pass &= check(ensemble(wl).logf == 0.5 * logf0, "logf halved\n")

    return pass
end

@testset "Wang-Landau" begin
    @testset "accept and reset" begin
        @test test_wang_landau_accept_and_reset()
    end
    @testset "update mechanics" begin
        @test test_wang_landau_update_mechanics()
    end
end
