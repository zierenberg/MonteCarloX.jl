# here tests should be written for utility functions in src/Utils.jl
using MonteCarloX
using StatsBase

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

    # check missing API
    if verbose
        println("... hist[200] == missing")
    end
    pass &= ismissing(hist[200])

    if verbose
        println("... hist[200] = 3 throws error")
    end
    valid = true
    try
        hist[200] = 3
        valid = false
    catch e
        vlaid = true
    end
    pass &= valid

    return pass
end

function test_log_sum(;verbose = false)
    pass = true
    if verbose
        println("... test for numbers that can be exactly converted")
    end
    a = 2.0; A = exp(a)
    b = 3.0; B = exp(b)
    C = A + B
    c = log_sum(a,b)
    pass &= log(C) == c

    if verbose
        println("... test for integer types")
    end
    a = 5; A = exp(a)
    b = 5; B = exp(b)
    C = A + B
    c = log_sum(a,b)
    pass &= log(C) == c

    if verbose
        println("... test for mixed types")
    end
    a = 5;   A = exp(a)
    b = 3.0; B = exp(b)
    C = A + B
    c = log_sum(a,b)
    pass &= log(C) == c

    return pass
end

function test_binary_search(verbose=false)
    pass = true

    pass &= (binary_search([1.,2.,3.,4.],2.5)==3)
    pass &= (binary_search([1.,2.,2.2,2.6,15.0],2.5)==4)
    pass &= (binary_search([1,2,3,4],2)==2)

    return pass
end
