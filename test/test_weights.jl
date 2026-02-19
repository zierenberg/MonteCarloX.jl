using MonteCarloX
using StatsBase
using Test

function test_tabulated_logweight_basics(; verbose=false)
    pass = true

    bins = 0.0:1.0:4.0
    hist = Histogram((collect(bins),), zeros(Float64, length(bins) - 1))
    lw = TabulatedLogWeight(hist)

    lw[1.2] = 0.7
    pass &= lw[1.2] == 0.7

    lw[2] = -0.3
    pass &= lw[2] == -0.3

    rhs = fill(0.5, size(lw)...)
    lw2 = lw - rhs
    pass &= lw2[1.2] == lw[1.2] - 0.5
    pass &= lw2[2] == lw[2] - 0.5

    if verbose
        println("TabulatedLogWeight basics:")
        println("  lw[1.2] = $(lw[1.2])")
        println("  lw2[1.2] = $(lw2[1.2])")
    end

    return pass
end

function run_weights_testsets(; verbose=false)
    @testset "Weights" begin
        @testset "TabulatedLogWeight" begin
            @test test_tabulated_logweight_basics(verbose=verbose)
        end
    end
    return true
end
