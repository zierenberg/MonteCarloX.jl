using MonteCarloX
using Random
using Test

function test_rng_mutable_static_sequence()
    pass = true
    seed = 1000

    rng = MutableRandomNumbers(MersenneTwister(seed), 10)
    rng_MT = MersenneTwister(seed)

    for (r1, r2) in zip(rand(rng, 10), rand(rng_MT, 10))
        pass &= r1 == r2
    end
    pass = check(pass, "static sequence matches MersenneTwister\n")

    threw = try; rand(rng); false; catch; true; end
    pass &= check(threw, "static mode throws after exhaustion\n")

    return pass
end

function test_rng_mutable_dynamic_sequence_and_reset()
    pass = true
    seed = 1000

    rng = MutableRandomNumbers(MersenneTwister(seed), 10, mode=:dynamic)
    pass &= check(length(rng) == 10, "initial length == 10\n")

    rng_MT = MersenneTwister(seed)
    for (r1, r2) in zip(rand(rng, 20), rand(rng_MT, 20))
        pass &= r1 == r2
    end
    pass = check(pass, "dynamic sequence matches MersenneTwister\n")

    pass &= check(length(rng) == 20, "length grew to 20\n")

    reset!(rng)
    pass &= check(rng.index_current == 0, "reset! zeroes index\n")

    rand_number = rand(rng)
    pass &= check(rand_number == rng[1], "first draw after reset matches rng[1]\n")

    return pass
end

function test_rng_mutable_randexp_sequence()
    pass = true
    seed = 1000

    rng = MutableRandomNumbers(MersenneTwister(seed), 10, mode=:dynamic)
    rng_MT = MersenneTwister(seed)

    for (r1, r2) in zip(randexp(rng, 10), randexp(rng_MT, 10))
        pass &= r1 == r2
    end
    pass = check(pass, "randexp sequence matches MersenneTwister\n")

    return pass
end

function test_rng_mutable_default_initialization_modes()
    pass = true

    rng = MutableRandomNumbers(100)
    pass &= check(rng.mode == :static, "MutableRandomNumbers(N) defaults to static\n")

    rng = MutableRandomNumbers()
    pass &= check(rng.mode == :dynamic, "MutableRandomNumbers() defaults to dynamic\n")

    return pass
end

function test_rng_mutable_dynamic_constructor()
    pass = true
    seed = 1000

    rng_base = MersenneTwister(seed)
    rng = MutableRandomNumbers(rng_base; mode=:dynamic)
    pass &= check(rng.mode == :dynamic, "constructor with rng arg is dynamic\n")
    pass &= check(length(rng) == 0, "initial length == 0\n")

    rng_MT = MersenneTwister(seed)
    pass &= check(rand(rng) == rand(rng_MT), "first draw matches MersenneTwister\n")

    return pass
end

function test_rng_rand_inbounds_closeopen01_mapping()
    pass = true
    seed = 1000

    rng = MutableRandomNumbers(MersenneTwister(seed), 2)
    reset!(rng)
    r01 = MonteCarloX.rand_inbounds(rng, Random.CloseOpen01())
    pass &= check(r01 == rng[1], "CloseOpen01 draw == rng[1]\n")

    reset!(rng)
    r12 = MonteCarloX.rand_inbounds(rng, Random.CloseOpen12())
    pass &= check(r01 == r12 - 1.0, "CloseOpen01 == CloseOpen12 - 1.0\n")

    return pass
end

function test_rng_mutable_access_and_manipulation()
    pass = true
    seed = 1000

    rng = MutableRandomNumbers(MersenneTwister(seed), 100)
    idx = 10
    backup = rng[idx]
    pass &= check(rng[idx] == backup, "getindex consistent\n")

    new = 0.23
    rng[idx] = new
    pass &= check(rng[idx] == new, "setindex! works\n")

    rng[idx] = 0
    pass &= check(rng[idx] == 0.0, "setindex! to 0 works\n")

    for bad_val in [1.1, 1.0, -0.1]
        threw = try; rng[1] = bad_val; false; catch; true; end
        pass &= check(threw, "setindex! rejects $bad_val\n")
    end

    return pass
end

@testset "RNG" begin
    @testset "Static sequence" begin
        @test test_rng_mutable_static_sequence()
    end
    @testset "Dynamic sequence and reset" begin
        @test test_rng_mutable_dynamic_sequence_and_reset()
    end
    @testset "Randexp sequence" begin
        @test test_rng_mutable_randexp_sequence()
    end
    @testset "Default initialization modes" begin
        @test test_rng_mutable_default_initialization_modes()
    end
    @testset "Dynamic constructor" begin
        @test test_rng_mutable_dynamic_constructor()
    end
    @testset "rand_inbounds CloseOpen01 mapping" begin
        @test test_rng_rand_inbounds_closeopen01_mapping()
    end
    @testset "Access and manipulation" begin
        @test test_rng_mutable_access_and_manipulation()
    end
end
