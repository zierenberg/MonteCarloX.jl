using MonteCarloX
import MonteCarloX: logistic
using StatsBase
using Test

function test_histogram_set_get()
    pass = true

    list_vals = [i for i = 1:100]
    hist = fit(Histogram, list_vals, 1:10:101)

    # set each bin via a coordinate inside it
    target = [i for i = 1:10]
    for (idx, t) in zip(1:10:100, target)
        hist[idx] = t
    end

    # verify all coordinates map to correct bin value
    for i = 1:100
        pass &= hist[i] == target[1 + floor(Int, (i - 1) / 10)]
    end
    pass = check(pass, "histogram set/get round-trip\n")

    pass &= check(ismissing(hist[200]), "out-of-bounds returns missing\n")

    threw = try; hist[200] = 3; false; catch; true; end
    pass &= check(threw, "out-of-bounds setindex! throws\n")

    return pass
end

function test_log_sum()
    pass = true

    a = 2.0; b = 3.0
    pass &= check(log(exp(a) + exp(b)) == log_sum(a, b), "log_sum(2.0, 3.0)\n")

    a = 5; b = 5
    pass &= check(log(exp(a) + exp(b)) == log_sum(a, b), "log_sum equal args\n")

    a = 5; b = 3.0
    pass &= check(log(exp(a) + exp(b)) == log_sum(a, b), "log_sum mixed types\n")

    vals = [1.2, -3.5, 0.7, -10.0]
    pass &= check(isapprox(log_sum(vals), log(sum(exp.(vals))); atol=1e-12), "log_sum(vector)\n")

    big_a = BigFloat("1.234567890123456789")
    big_b = BigFloat("1.234567890123456788")
    big_c = log_sum(big_a, big_b)
    pass &= check(big_c isa BigFloat, "log_sum BigFloat type\n")
    pass &= check(isapprox(big_c, log(exp(big_a) + exp(big_b)); rtol=big(1e-30)), "log_sum BigFloat precision\n")

    return pass
end

function test_logistic()
    pass = true

    pass &= check(logistic(0.0) == 0.5, "logistic(0.0) == 0.5\n")
    pass &= check(isapprox(logistic(20.0), 1.0; atol=1e-8), "logistic(20.0) ≈ 1.0\n")
    pass &= check(isapprox(logistic(-20.0), 0.0; atol=1e-8), "logistic(-20.0) ≈ 0.0\n")

    x = 3.7
    pass &= check(isapprox(logistic(x) + logistic(-x), 1.0; atol=1e-12), "logistic symmetry\n")

    return pass
end

function test_binary_search()
    pass = true

    pass &= check(binary_search([1.,2.,3.,4.], 2.5) == 3, "binary_search float\n")
    pass &= check(binary_search([1.,2.,2.2,2.6,15.0], 2.5) == 4, "binary_search float 2\n")
    pass &= check(binary_search([1,2,3,4], 2) == 2, "binary_search int\n")

    return pass
end

@testset "Utils" begin
    @testset "Histogram set/get" begin
        @test test_histogram_set_get()
    end
    @testset "log_sum" begin
        @test test_log_sum()
    end
    @testset "logistic" begin
        @test test_logistic()
    end
    @testset "binary_search" begin
        @test test_binary_search()
    end
end
