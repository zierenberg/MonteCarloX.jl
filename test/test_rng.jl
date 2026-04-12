using MonteCarloX
using Random
using Test

function test_mutable_rng_constructors()
    pass = true

    # MutableRandomNumbers(N) defaults to static
    rng_static = MutableRandomNumbers(100)
    pass &= check(rng_static.mode == :static, "MutableRandomNumbers(N) defaults to static\n")

    # MutableRandomNumbers() defaults to dynamic
    rng_dynamic = MutableRandomNumbers()
    pass &= check(rng_dynamic.mode == :dynamic, "MutableRandomNumbers() defaults to dynamic\n")

    # from existing rng with dynamic mode, starts empty
    seed = 1000
    rng = MutableRandomNumbers(MersenneTwister(seed); mode=:dynamic)
    pass &= check(rng.mode == :dynamic, "constructor with rng arg is dynamic\n")
    pass &= check(length(rng) == 0, "initial length == 0\n")
    pass &= check(rand(rng) == rand(MersenneTwister(seed)), "first draw matches MersenneTwister\n")

    return pass
end

function test_mutable_rng_sequences()
    pass = true
    seed = 1000

    # static mode: matches MersenneTwister, throws after exhaustion
    rng_s = MutableRandomNumbers(MersenneTwister(seed), 10)
    rng_ref = MersenneTwister(seed)
    for (r1, r2) in zip(rand(rng_s, 10), rand(rng_ref, 10))
        pass &= r1 == r2
    end
    pass = check(pass, "static sequence matches MersenneTwister\n")

    threw = try; rand(rng_s); false; catch; true; end
    pass &= check(threw, "static mode throws after exhaustion\n")

    # dynamic mode: matches MersenneTwister, length grows, reset works
    rng_d = MutableRandomNumbers(MersenneTwister(seed), 10, mode=:dynamic)
    rng_ref = MersenneTwister(seed)
    for (r1, r2) in zip(rand(rng_d, 20), rand(rng_ref, 20))
        pass &= r1 == r2
    end
    pass = check(pass, "dynamic sequence matches MersenneTwister\n")
    pass &= check(length(rng_d) == 20, "length grew to 20\n")

    reset!(rng_d)
    pass &= check(rng_d.index_current == 0, "reset! zeroes index\n")
    pass &= check(rand(rng_d) == rng_d[1], "first draw after reset matches rng_d[1]\n")

    # randexp matches MersenneTwister
    rng_e = MutableRandomNumbers(MersenneTwister(seed), 10, mode=:dynamic)
    rng_ref = MersenneTwister(seed)
    for (r1, r2) in zip(randexp(rng_e, 10), randexp(rng_ref, 10))
        pass &= r1 == r2
    end
    pass = check(pass, "randexp sequence matches MersenneTwister\n")

    return pass
end

function test_mutable_rng_access()
    pass = true
    seed = 1000

    # getindex / setindex!
    rng = MutableRandomNumbers(MersenneTwister(seed), 100)
    idx = 10
    backup = rng[idx]
    pass &= check(rng[idx] == backup, "getindex consistent\n")

    rng[idx] = 0.23
    pass &= check(rng[idx] == 0.23, "setindex! works\n")

    rng[idx] = 0
    pass &= check(rng[idx] == 0.0, "setindex! to 0 works\n")

    for bad_val in [1.1, 1.0, -0.1]
        threw = try; rng[1] = bad_val; false; catch; true; end
        pass &= check(threw, "setindex! rejects $bad_val\n")
    end

    # rand_inbounds with CloseOpen01 and CloseOpen12
    rng2 = MutableRandomNumbers(MersenneTwister(seed), 2)
    reset!(rng2)
    r01 = MonteCarloX.rand_inbounds(rng2, Random.CloseOpen01())
    pass &= check(r01 == rng2[1], "CloseOpen01 draw == rng2[1]\n")

    reset!(rng2)
    r12 = MonteCarloX.rand_inbounds(rng2, Random.CloseOpen12())
    pass &= check(r01 == r12 - 1.0, "CloseOpen01 == CloseOpen12 - 1.0\n")

    return pass
end

@testset "RNG" begin
    @testset "constructors" begin
        @test test_mutable_rng_constructors()
    end
    @testset "sequences" begin
        @test test_mutable_rng_sequences()
    end
    @testset "access and manipulation" begin
        @test test_mutable_rng_access()
    end
end
