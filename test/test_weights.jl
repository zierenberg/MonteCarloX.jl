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

function test_tabulated_logweight_properties(; verbose=false)
    pass = true

    edges = [0.0, 1.0, 2.0]
    lw = TabulatedLogWeight(edges, -1.0)

    pass &= propertynames(lw) == (:histogram, :table)
    pass &= propertynames(lw, true) == (:histogram, :table)

    new_hist = Histogram((collect(edges),), fill(0.5, length(edges) - 1))
    lw.table = new_hist
    pass &= lw.histogram === new_hist
    pass &= lw.table === new_hist

    pass &= size(lw) == size(new_hist.weights)

    pass &= lw[1] == 0.5
    lw[1] = 1.25
    pass &= lw[1] == 1.25

    if verbose
        println("TabulatedLogWeight properties test pass: $(pass)")
    end

    return pass
end

function test_tabulated_logweight_invalid_edges(; verbose=false)
    pass = true

    valid = false
    try
        TabulatedLogWeight([0.0], 0.0)
    catch err
        valid = err isa ArgumentError
    end
    pass &= valid

    if verbose
        println("TabulatedLogWeight invalid edges test pass: $(pass)")
    end

    return pass
end

function run_weights_testsets(; verbose=false)
    @testset "Weights" begin
        @testset "TabulatedLogWeight" begin
            @test test_tabulated_logweight_basics(verbose=verbose)
            @test test_tabulated_logweight_properties(verbose=verbose)
            @test test_tabulated_logweight_invalid_edges(verbose=verbose)
        end
    end
    return true
end
