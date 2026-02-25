using MonteCarloX
using StatsBase
using Test

# test for BinnedLogWeight
function test_binned_logweight_discrete(; verbose=false)
    pass = true 

    # 1D
    bins = 0:2:10
    lw = BinnedLogWeight(bins, 0.0)
    pass &= lw isa BinnedLogWeight{1,MonteCarloX.DiscreteBinning{Int64}}
    pass &= all(iszero, lw.weights)
    pass &= size(lw) == (6,)
    pass &= size(lw.weights) == (6,)
    pass &= lw.bins[1].start == 0
    pass &= lw.bins[1].step == 2

    binvals = collect(lw.bins[1])
    pass &= length(binvals) == 6
    pass &= binvals == [0, 2, 4, 6, 8, 10]

    lw[4] = 1.5
    pass &= lw[4] == 1.5
    pass &= lw(4) == lw[4]
    lw[6] = -0.5
    pass &= lw(6) == -0.5

    # test out of bounds
    valid = false
    try
        lw[11] = 0.0
    catch err
        valid = err isa BoundsError
    end
    try
        lw[-1] = 0.0
    catch err
        valid &= err isa BoundsError
    end

    # test _assert_same_domain
    lw2 = BinnedLogWeight(0:2:10, 0.0)
    pass &= MonteCarloX._assert_same_domain(lw, lw2) == nothing
    lw3 = BinnedLogWeight(0:1:10, 0.0)
    valid = false
    try
        MonteCarloX._assert_same_domain(lw, lw3)
    catch err
        valid = err isa AssertionError
    end
    pass &= valid

    if verbose
        println("BinnedLogWeight discrete 1D test pass: $(pass)")
    end

    # 2D
    bins2d = (0:1:5, 0:2:10)
    lw2d = BinnedLogWeight(bins2d, 0.0)
    pass &= lw2d isa BinnedLogWeight{2, MonteCarloX.DiscreteBinning{Int64}}
    pass &= all(iszero, lw2d.weights)
    pass &= size(lw2d) == (6, 6)
    pass &= size(lw2d.weights) == (6, 6)
    pass &= lw2d.bins[1].start == 0 
    pass &= lw2d.bins[2].start == 0 
    pass &= lw2d.bins[1].step == 1
    pass &= lw2d.bins[2].step == 2

    lw2d[3, 4] = -0.5
    pass &= lw2d[3, 4] == -0.5
    pass &= lw2d(3, 4) == lw2d[3, 4]

    # return test result
    if verbose
        println("BinnedLogWeight discrete 2D test pass: $(pass)")
    end

    return pass
end

function test_binned_logweight_continuous(; verbose=false)
    pass = true 

    # 1D
    edges = 0.0:1.0:4.0
    lw = BinnedLogWeight(edges, 0.0)
    pass &= lw isa BinnedLogWeight{1,MonteCarloX.ContinuousBinning{Float64}}
    pass &= all(iszero, lw.weights)
    pass &= size(lw) == (4,)
    pass &= size(lw.weights) == (4,)
    pass &= size(lw.bins[1].edges) == (5,)
    pass &= size(lw.bins[1].centers) == (4,)
    pass &= lw.bins[1].edges == edges
    pass &= lw.bins[1].centers == [0.5, 1.5, 2.5, 3.5]

    lw[1.2] = 1.5
    pass &= lw.weights[2] == 1.5 
    pass &= lw[1.2] == 1.5
    pass &= lw(1.2) == lw[1.2]

    binvals = collect(lw.bins[1])
    pass &= length(binvals) == length(edges) - 1

    # out-of-bounds for continous are stored at the boundaries
    # valid = false
    # try
    #     lw[4.5] = 0.0
    #     valid = false # should not be reached if out of bounds is properly handled
    #     @warn "Expected BoundsError but none was raised"
    # catch err
    #     valid = err isa BoundsError
    #     @debug "Caught expected error: $err"
    # end
    # try
    #     lw[-0.5] = 0.0
    #     valid &= false # should not be reached if out of bounds is properly handled
    #     @warn "Expected BoundsError but none was raised"
    # catch err
    #     valid &= err isa BoundsError
    #     @debug "Caught expected error: $err"
    # end

    # test _assert_same_domain
    lw2 = BinnedLogWeight(edges, 0.0)
    pass &= MonteCarloX._assert_same_domain(lw, lw2) == nothing
    lw3 = BinnedLogWeight(0.0:0.5:4.0, 0.0)
    valid=false
    try        
        MonteCarloX._assert_same_domain(lw, lw3)
        @warn "Expected AssertionError but none was raised"
    catch err
        valid = err isa AssertionError
    end
    pass &= valid
    lw_discrete = BinnedLogWeight(0:1:4, 0.0)
    valid=false
    try
        MonteCarloX._assert_same_domain(lw, lw_discrete)
        @warn "Expected AssertionError but none was raised"
    catch err
        valid = err isa AssertionError
    end 
    pass &= valid

    if verbose
        println("BinnedLogWeight continuous 1D test pass: $(pass)")
    end

    # 2D
    edges2d = (0.0:1.0:3.0, 0.0:2.0:6.0)
    lw2d = BinnedLogWeight(edges2d, 0.0)
    pass &= lw2d isa BinnedLogWeight{2,MonteCarloX.ContinuousBinning{Float64}}
    pass &= all(iszero, lw2d.weights)
    pass &= size(lw2d) == (3, 3)
    pass &= size(lw2d.weights) == (3, 3)
    pass &= size(lw2d.bins[1].edges) == (4,)
    pass &= size(lw2d.bins[2].edges) == (4,) 
    pass &= size(lw2d.bins[1].centers) == (3,)
    pass &= size(lw2d.bins[2].centers) == (3,)
    pass &= lw2d.bins[1].edges == edges2d[1]
    pass &= lw2d.bins[2].edges == edges2d[2]
    pass &= lw2d.bins[1].centers == [0.5, 1.5, 2.5]
    pass &= lw2d.bins[2].centers == [1.0, 3.0, 5.0]

    lw2d[1.2, 2.5] -= 0.5
    pass &= lw2d.weights[2,2] == -0.5 
    pass &= lw2d[1.2, 2.5] == -0.5
    pass &= lw2d(1.2, 2.5) == lw2d[1.2, 2.5]

    # return test result
    if verbose
        println("BinnedLogWeight continuous 2D test pass: $(pass)") 
    end

    return pass
end

function run_weights_testsets(; verbose=false)
    @testset "Weights" begin
        @testset "BinnedLogWeight" begin
            @test test_binned_logweight_discrete(verbose=verbose)
            @test test_binned_logweight_continuous(verbose=verbose)
        end
    end
    return true
end
