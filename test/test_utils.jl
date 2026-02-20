# here tests should be written for utility functions in src/Utils.jl
using MonteCarloX
using StatsBase
using Test

function test_histogram_set_get(; verbose=false)
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
        pass &= h == t
    end

    # check missing API
    pass &= ismissing(hist[200])

    valid = true
    try
        hist[200] = 3
        valid = false
    catch
        valid = true
    end
    pass &= valid

    if verbose
        println("Histogram set/get test pass: $(pass)")
    end

    return pass
end

function test_log_sum(; verbose=false)
    pass = true
    a = 2.0; A = exp(a)
    b = 3.0; B = exp(b)
    C = A + B
    c = log_sum(a,b)
    pass &= log(C) == c

    a = 5; A = exp(a)
    b = 5; B = exp(b)
    C = A + B
    c = log_sum(a,b)
    pass &= log(C) == c

    a = 5;   A = exp(a)
    b = 3.0; B = exp(b)
    C = A + B
    c = log_sum(a,b)
    pass &= log(C) == c

    if verbose
        println("log_sum test pass: $(pass)")
    end

    return pass
end

function test_logistic(; verbose=false)
    pass = true

    pass &= logistic(0.0) == 0.5

    pass &= isapprox(logistic(5.0), 1.0; atol = 1e-6)
    pass &= isapprox(logistic(-5.0), 0.0; atol = 1e-6)

    x = 3.7
    pass &= isapprox(logistic(x) + logistic(-x), 1.0; atol = 1e-12)

    if verbose
        println("logistic test pass: $(pass)")
    end

    return pass
end

function test_binary_search(; verbose=false)
    pass = true

    pass &= (binary_search([1.,2.,3.,4.],2.5)==3)
    pass &= (binary_search([1.,2.,2.2,2.6,15.0],2.5)==4)
    pass &= (binary_search([1,2,3,4],2)==2)

    if verbose
        println("binary_search test pass: $(pass)")
    end

    return pass
end

function run_utils_testsets(; verbose=false)
    @testset "Utils" begin
        @testset "Histogram set/get" begin
            @test test_histogram_set_get(verbose=verbose)
        end
        @testset "log_sum" begin
            @test test_log_sum(verbose=verbose)
        end
        @testset "logistic" begin
            @test test_logistic(verbose=verbose)
        end
        @testset "binary_search" begin
            @test test_binary_search(verbose=verbose)
        end
    end
    return true
end
