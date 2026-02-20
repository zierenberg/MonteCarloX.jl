using MonteCarloX
using StatsBase
using Test

function test_tabulated_logweight_basics(; verbose=false)
    pass = true

    bins = 0.0:1.0:4.0
    lw_init = TabulatedLogWeight(bins, 0.0)
    pass &= lw_init isa TabulatedLogWeight
    pass &= size(lw_init) == (4,)
    pass &= all(iszero, lw_init.histogram.weights)

    hist = Histogram((collect(bins),), zeros(Float64, length(bins) - 1))
    lw = TabulatedLogWeight(hist)

    lw[1.2] = 0.7
    pass &= lw[1.2] == 0.7
    pass &= lw(1.2) == lw[1.2]

    lw[2] = -0.3
    pass &= lw[2] == -0.3

    pass &= lw.histogram === hist
    pass &= lw.table === hist

    if verbose
        println("TabulatedLogWeight basics:")
        println("  lw[1.2] = $(lw[1.2])")
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
