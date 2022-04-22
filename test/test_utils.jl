# here tests should be written for utility functions in src/Utils.jl
using MonteCarloX
using StatsBase

include("./includes.jl")

function test_histogram_set_get()
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
        h = hist[i]
        t = target[1 + floor(Int, (i - 1) / 10)]
        pass &= check(h == t, @sprintf("... check bin for %d: %d == %d\n", i, h, t))
    end

    # check missing API
    pass &= check(ismissing(hist[200]), "... hist[200] == missing\n")

    valid = true
    try
        hist[200] = 3
        valid = false
    catch e
        vlaid = true
    end
    pass &= check(valid, "... hist[200] = 3 throws error\n")

    return pass
end

function test_log_sum()
    pass = true
    a = 2.0; A = exp(a)
    b = 3.0; B = exp(b)
    C = A + B
    c = log_sum(a,b)
    pass &= check(log(C) == c, "... test float types\n")

    a = 5; A = exp(a)
    b = 5; B = exp(b)
    C = A + B
    c = log_sum(a,b)
    pass &= check(log(C) == c, "... test integer types\n")

    a = 5;   A = exp(a)
    b = 3.0; B = exp(b)
    C = A + B
    c = log_sum(a,b)
    pass &= check(log(C) == c, "... test mixed types\n")

    return pass
end

function test_binary_search()
    pass = true

    pass &= (binary_search([1.,2.,3.,4.],2.5)==3)
    pass &= (binary_search([1.,2.,2.2,2.6,15.0],2.5)==4)
    pass &= (binary_search([1,2,3,4],2)==2)

    return pass
end
