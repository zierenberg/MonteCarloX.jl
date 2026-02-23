using MonteCarloX
using Random
using Test

function test_rng_mutable_static_sequence(; verbose=false)
    pass = true
    seed = 1000

    if verbose
        println("\nRNG mutable (static mode)")
    end
    rng = MutableRandomNumbers(MersenneTwister(seed),10)
    rng_MT = MersenneTwister(seed)

    if verbose
        println("  checking uniform draws")
    end
    for (r1, r2) in zip(rand(rng, 10), rand(rng_MT,10))
        pass &= r1 == r2
    end
    valid = true
    try
        rand(rng)
        valid = false
    catch
        valid = true
    end
    pass &= valid

    return pass
end

function test_rng_mutable_dynamic_sequence_and_reset(; verbose=false)
    pass = true
    seed = 1000


    if verbose
        println("\nRNG mutable (dynamic mode)")
    end
    rng = MutableRandomNumbers(MersenneTwister(seed),10, mode=:dynamic)
    pass &= length(rng) == 10
    rng_MT = MersenneTwister(seed)
    for (r1, r2) in zip(rand(rng, 20), rand(rng_MT,20))
        pass &= r1 == r2
    end
    length_final = length(rng)
    pass &= length_final == 20
    reset!(rng)
    pass &= rng.index_current == 0

    rand_number = rand(rng)
    pass &= rand_number == rng[1]

    return pass
end

function test_rng_mutable_randexp_sequence(; verbose=false)
    pass = true
    seed = 1000


    # randexp requires to redraw random number in 1.1% of tries ... requires mode=:dynamic
    rng = MutableRandomNumbers(MersenneTwister(seed),10, mode=:dynamic)
    rng_MT = MersenneTwister(seed)
    if verbose
        println("  checking randexp sequence")
    end
    for (r1, r2) in zip(randexp(rng, 10), randexp(rng_MT,10))
        pass &= r1 == r2
    end

    return pass
end

function test_rng_mutable_default_initialization_modes(; verbose=false)
    pass = true


    if verbose
        println("\nRNG default initialization modes")
    end
    rng = MutableRandomNumbers(100)
    pass &= rng.mode == :static
    rng = MutableRandomNumbers()
    pass &= rng.mode == :dynamic

    return pass
end

function test_rng_mutable_dynamic_constructor(; verbose=false)
    pass = true
    seed = 1000

    if verbose
        println("\nRNG dynamic constructor with rng argument")
    end
    rng_base = MersenneTwister(seed)
    rng = MutableRandomNumbers(rng_base; mode=:dynamic)
    pass &= rng.mode == :dynamic
    pass &= length(rng) == 0
    rng_MT = MersenneTwister(seed)
    r1 = rand(rng)
    r2 = rand(rng_MT)
    pass &= r1 == r2

    return pass
end

function test_rng_rand_inbounds_closeopen01_mapping(; verbose=false)
    pass = true
    seed = 1000

    if verbose
        println("\nRNG rand_inbounds mapping")
    end
    rng = MutableRandomNumbers(MersenneTwister(seed), 2)
    reset!(rng)
    r01 = MonteCarloX.rand_inbounds(rng, Random.CloseOpen01())
    pass &= r01 == rng[1]
    reset!(rng)
    r12 = MonteCarloX.rand_inbounds(rng, Random.CloseOpen12())
    pass &= r01 == r12 - 1.0

    return pass
end

function test_rng_mutable_access_and_manipulation(; verbose=false)
    pass = true
    seed = 1000




    if verbose
        println("\nRNG access and manipulation")
    end
    rng = MutableRandomNumbers(MersenneTwister(seed),100)
    idx = 10
    backup = rng[idx]
    pass &= rng[idx] == backup
    new = 0.23
    rng[idx] = new
    pass &= rng[idx] == new
    rng[idx] = 0
    pass &= rng[idx] == 0.0
    valid = true
    try
        rng[1] = 1.1
        valid = false
    catch
        valid = true
    end
    pass &= valid
    valid = true
    try
        rng[1] = 1.0
        valid = false
    catch
        valid = true
    end
    pass &= valid
    valid = true
    try
        rng[1] = -0.1
        valid = false
    catch
        valid = true
    end
    pass &= valid

    return pass
end

function test_rng_mutable(; verbose=false)
    pass = true
    pass &= test_rng_mutable_static_sequence(verbose=verbose)
    pass &= test_rng_mutable_dynamic_sequence_and_reset(verbose=verbose)
    pass &= test_rng_mutable_randexp_sequence(verbose=verbose)
    pass &= test_rng_mutable_default_initialization_modes(verbose=verbose)
    pass &= test_rng_mutable_dynamic_constructor(verbose=verbose)
    pass &= test_rng_rand_inbounds_closeopen01_mapping(verbose=verbose)
    pass &= test_rng_mutable_access_and_manipulation(verbose=verbose)
    return pass
end

function run_rng_testsets(; verbose=false)
    @testset "RNG" begin
        @testset "Static sequence" begin
            @test test_rng_mutable_static_sequence(verbose=verbose)
        end
        @testset "Dynamic sequence and reset" begin
            @test test_rng_mutable_dynamic_sequence_and_reset(verbose=verbose)
        end
        @testset "Randexp sequence" begin
            @test test_rng_mutable_randexp_sequence(verbose=verbose)
        end
        @testset "Default initialization modes" begin
            @test test_rng_mutable_default_initialization_modes(verbose=verbose)
        end
        @testset "Dynamic constructor" begin
            @test test_rng_mutable_dynamic_constructor(verbose=verbose)
        end
        @testset "rand_inbounds CloseOpen01 mapping" begin
            @test test_rng_rand_inbounds_closeopen01_mapping(verbose=verbose)
        end
        @testset "Access and manipulation" begin
            @test test_rng_mutable_access_and_manipulation(verbose=verbose)
        end
    end
    return true
end


