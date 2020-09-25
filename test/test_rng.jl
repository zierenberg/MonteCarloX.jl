using MonteCarloX
using StatsBase

function test_rng_mutable(;verbose = false)
    pass = true
    seed = 1000

    if verbose; println("\nstatic mutable random numbers"); end
    rng = MutableRandomNumbers(MersenneTwister(seed),100)
    rng_MT = MersenneTwister(seed)

    if verbose; println("uniform floats:"); end
    for (r1, r2) in zip(rand(rng, 10), rand(rng_MT,10))
        if verbose; println(r1, " vs ", r2); end 
        pass &= r1==r2
    end

    if verbose; println("\nexponentially distributed floats:"); end
    for (r1, r2) in zip(randexp(rng, 10), randexp(rng_MT,10))
        if verbose; println(r1, " vs ", r2); end
        pass &= r1==r2
    end

    if verbose; println("\ndynamic mutable random numbers"); end
    rng = MutableRandomNumbers(MersenneTwister(seed),10, mode=:dynamic)
    rng_MT = MersenneTwister(seed)
    for (i, r1, r2) in zip(1:20, rand(rng, 20), rand(rng_MT,20))
        status = "old"
        if i > 10; status = "new"; end
        if verbose; println(status, " ", r1, " vs ", r2); end
        pass &= r1==r2
    end

    if verbose; println("\ntest outcome:"); end
    return pass
end

function test_histogram_set_get(;verbose = false)
    pass = true

    # bin uniform values into histogram (each bin has 10 elements)
    list_vals = [i for i = 1:100]
    hist = fit(Histogram, list_vals, 1:10:101)

    # reset the size of the histogram bins (accessing each bin via random
    # elements that are included in the bin)
    target = [i for i = 1:10]
    hist[1] = target[1]
    hist[12] = target[2]
    hist[23] = target[3]
    hist[34] = target[4]
    hist[45] = target[5]
    hist[56] = target[6]
    hist[67] = target[7]
    hist[78] = target[8]
    hist[89] = target[9]
    hist[100] = target[10]

    # check that the histgram entries correspond to the target values
    for i = 1:100
        if verbose
            println("... $(hist[i]) == $(target[1 + floor(Int, (i - 1) / 10)])")
        end
        pass &= hist[i] == target[1 + floor(Int, (i - 1) / 10)]
    end

    return pass
end
