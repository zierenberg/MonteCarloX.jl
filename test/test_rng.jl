using MonteCarloX
using StatsBase
using Random

include("includes.jl")

function test_rng_mutable()
    pass = true
    seed = 1000

    if test_verbose; println("\ntest static mutable random numbers"); end
    rng = MutableRandomNumbers(MersenneTwister(seed),100)
    rng_MT = MersenneTwister(seed)

    if test_verbose; println("...test uniform floats:"); end
    for (r1, r2) in zip(rand(rng, 10), rand(rng_MT,10))
        pass &= check(r1==r2, @sprintf("... %f vs %f\n", r1, r2))
    end

    if test_verbose; println("...test exponentially distributed floats:"); end
    for (r1, r2) in zip(randexp(rng, 10), randexp(rng_MT,10))
        pass &= check(r1==r2, @sprintf("... %f vs %f\n", r1, r2))
    end

    if test_verbose; println("\ntest ndynamic mutable random numbers"); end
    rng = MutableRandomNumbers(MersenneTwister(seed),10, mode=:dynamic)
    pass &= length(rng)==10
    rng_MT = MersenneTwister(seed)
    for (i, r1, r2) in zip(1:20, rand(rng, 20), rand(rng_MT,20))
        status = "old"
        if i > 10; status = "new"; end
        pass &= check(r1==r2, @sprintf("... %s: %f vs %f\n", status, r1, r2))
    end
    length_final = length(rng)
    pass &= check(length_final==20,
                  @sprintf("...final length: %d vs 20\n", length_final))
    reset(rng)
    pass &= check(rng.index_current==0,
                  @sprintf("...reset rng shifts index to zero: %d\n",  rng.index_current))

    rand_number = rand(rng)
    pass &= check(rand_number == rng[1],
                  @sprintf("...such that next rand returns first number: %f vs %f\n",  rand_number, rng[1]))

    if test_verbose; println("\ntest default initialization modes"); end
    rng = MutableRandomNumbers(100)
    pass &= check(rng.mode == :static,
                  @sprintf("...default for empty is static: %s\n",rng.mode)
                 )
    rng = MutableRandomNumbers()
    pass &= check(rng.mode == :dynamic,
                  @sprintf("...default for empty is dynamic: %s\n",rng.mode)
                 )

    if test_verbose; println("\ntest access and manipulation"); end
    rng = MutableRandomNumbers(MersenneTwister(seed),100)
    idx = 10
    backup = rng[idx]
    pass &= check(rng[idx] == backup,
                  @sprintf("...old value at pos %d is %f vs %f\n",idx,rng[idx], backup))
    new = 0.23
    rng[idx] = new
    pass &= check(rng[idx] == new,
                  @sprintf("...set  value at pos %d as %f vs %f\n",idx,new, rng[idx]))

    if test_verbose; println("\ntest outcome:"); end
    return pass
end


