using MonteCarloX
using StatsBase
using Test

# test for BinnedObject
function test_binned_object_discrete(; verbose=false)
    pass = true 

    # 1D
    bins = 0:2:10
    bo = BinnedObject(bins) # default constructor with init=0.0
    pass &= bo isa BinnedObject{1,Float64,MonteCarloX.DiscreteBinning{Int64}}
    pass &= all(iszero, bo.values)
    pass &= size(bo) == (6,)
    pass &= size(bo.values) == (6,)
    pass &= bo.bins[1].start == 0
    pass &= bo.bins[1].step == 2

    binvals = get_centers(bo)
    pass &= length(binvals) == 6
    pass &= binvals == [0, 2, 4, 6, 8, 10]

    bo[4] = 1.5
    pass &= bo[4] == 1.5
    pass &= bo(4) == bo[4]
    bo[6] = -0.5
    pass &= bo(6) == -0.5

    pass &= !hasproperty(bo, :weights)

    # test out of bounds
    valid = false
    try
        bo[11] = 0.0
    catch err
        valid = err isa BoundsError
    end
    try
        bo[-1] = 0.0
    catch err
        valid &= err isa BoundsError
    end

    # test constructor with vector domain
    bins_vec = [0, 2, 4, 6, 8, 10]
    bo_vec = BinnedObject(bins_vec, 0.0)
    pass &= bo_vec isa BinnedObject{1,Float64,MonteCarloX.DiscreteBinning{Int64}}
    pass &= all(iszero, bo_vec.values)
    pass &= size(bo_vec) == (6,)
    pass &= size(bo_vec.values) == (6,)
    pass &= bo_vec.bins[1].start == 0
    pass &= bo_vec.bins[1].step == 2
    pass &= get_centers(bo_vec) == bins_vec

    # test special case of single-bin vector domain to throw an error
    valid = false
    try
        BinnedObject([0], 0.0)
    catch err
        valid = err isa ArgumentError
    end
    pass &= valid

    # error for non-equidistant bins
    valid = false
    try      
        BinnedObject([0, 1, 3], 0.0)
    catch err
        valid = err isa ArgumentError
    end
    pass &= valid

    # error when domain types is not supported
    valid = false
    try
        BinnedObject("invalid domain", 0.0)
    catch err
        valid = err isa ArgumentError
    end
    pass &= valid

    # test non-integer DiscreteBinning (even though not used by the Framework)
    bins_float = 0.0:2.0:10.0
    b = MonteCarloX.DiscreteBinning(first(bins_float), step(bins_float), length(bins_float))
    pass &= b isa MonteCarloX.DiscreteBinning{Float64}
    pass &= get_centers(b) == collect(bins_float)
    pass &= MonteCarloX._binindex(b, 4.0) == 3

    # test _assert_same_domain
    bo2 = BinnedObject(0:2:10, 0.0)
    pass &= MonteCarloX._assert_same_domain(bo, bo2) == nothing
    bo3 = BinnedObject(0:1:10, 0.0)
    valid = false
    try
        MonteCarloX._assert_same_domain(bo, bo3)
    catch err
        valid = err isa AssertionError
    end
    pass &= valid

    if verbose
        println("BinnedObject discrete 1D test pass: $(pass)")
    end

    # 2D
    bins2d = (0:1:5, 0:2:10)
    bo2d = BinnedObject(bins2d, 0.0)
    pass &= bo2d isa BinnedObject{2, Float64, MonteCarloX.DiscreteBinning{Int64}}
    pass &= all(iszero, bo2d.values)
    pass &= size(bo2d) == (6, 6)
    pass &= size(bo2d.values) == (6, 6)
    pass &= bo2d.bins[1].start == 0 
    pass &= bo2d.bins[2].start == 0 
    pass &= bo2d.bins[1].step == 1
    pass &= bo2d.bins[2].step == 2

    bo2d[3, 4] = -0.5
    pass &= bo2d[3, 4] == -0.5
    pass &= bo2d(3, 4) == bo2d[3, 4]

    pass &= !hasproperty(bo2d, :weights)

    # return test result
    if verbose
        println("BinnedObject discrete 2D test pass: $(pass)")
    end

    return pass
end

function test_binned_object_continuous(; verbose=false)
    pass = true 

    # 1D
    edges = 0.0:1.0:4.0
    bo = BinnedObject(edges, 0.0)
    pass &= bo isa BinnedObject{1,Float64,MonteCarloX.ContinuousBinning{Float64}}
    pass &= all(iszero, bo.values)
    pass &= size(bo) == (4,)
    pass &= size(bo.values) == (4,)
    pass &= size(bo.bins[1].edges) == (5,)
    pass &= size(bo.bins[1].centers) == (4,)
    pass &= bo.bins[1].edges == edges
    pass &= bo.bins[1].centers == [0.5, 1.5, 2.5, 3.5]

    # float ranges are interpreted as continuous edges by default;
    # explicit mode can force discrete center interpretation.
    bo_discrete_float = BinnedObject(edges, 0.0; interpretation=:discrete)
    pass &= bo_discrete_float isa BinnedObject{1,Float64,MonteCarloX.DiscreteBinning{Float64}}
    pass &= get_centers(bo_discrete_float) == collect(edges)

    bo[1.2] = 1.5
    pass &= bo.values[2] == 1.5 
    pass &= bo[1.2] == 1.5
    pass &= bo(1.2) == bo[1.2]

    pass &= !hasproperty(bo, :weights)

    binvals = get_centers(bo)
    pass &= length(binvals) == length(edges) - 1

    # out-of-bounds for continous are stored at the boundaries
    # valid = false
    # try
    #     bo[4.5] = 0.0
    #     valid = false # should not be reached if out of bounds is properly handled
    #     @warn "Expected BoundsError but none was raised"
    # catch err
    #     valid = err isa BoundsError
    #     @debug "Caught expected error: $err"
    # end
    # try
    #     bo[-0.5] = 0.0
    #     valid &= false # should not be reached if out of bounds is properly handled
    #     @warn "Expected BoundsError but none was raised"
    # catch err
    #     valid &= err isa BoundsError
    #     @debug "Caught expected error: $err"
    # end

    # test _assert_same_domain
    bo2 = BinnedObject(edges, 0.0)
    pass &= MonteCarloX._assert_same_domain(bo, bo2) == nothing
    bo3 = BinnedObject(0.0:0.5:4.0, 0.0)
    valid=false
    try        
        MonteCarloX._assert_same_domain(bo, bo3)
        @warn "Expected AssertionError but none was raised"
    catch err
        valid = err isa AssertionError
    end
    pass &= valid
    bo_discrete = BinnedObject(0:1:4, 0.0)
    valid=false
    try
        MonteCarloX._assert_same_domain(bo, bo_discrete)
        @warn "Expected AssertionError but none was raised"
    catch err
        valid = err isa AssertionError
    end 
    pass &= valid

    if verbose
        println("BinnedObject continuous 1D test pass: $(pass)")
    end

    # 2D
    edges2d = (0.0:1.0:3.0, 0.0:2.0:6.0)
    bo2d = BinnedObject(edges2d, 0.0)
    pass &= bo2d isa BinnedObject{2,Float64,MonteCarloX.ContinuousBinning{Float64}}
    pass &= all(iszero, bo2d.values)
    pass &= size(bo2d) == (3, 3)
    pass &= size(bo2d.values) == (3, 3)
    pass &= size(bo2d.bins[1].edges) == (4,)
    pass &= size(bo2d.bins[2].edges) == (4,) 
    pass &= size(bo2d.bins[1].centers) == (3,)
    pass &= size(bo2d.bins[2].centers) == (3,)
    pass &= bo2d.bins[1].edges == edges2d[1]
    pass &= bo2d.bins[2].edges == edges2d[2]
    pass &= bo2d.bins[1].centers == [0.5, 1.5, 2.5]
    pass &= bo2d.bins[2].centers == [1.0, 3.0, 5.0]

    bo2d[1.2, 2.5] -= 0.5
    pass &= bo2d.values[2,2] == -0.5 
    pass &= bo2d[1.2, 2.5] == -0.5
    pass &= bo2d(1.2, 2.5) == bo2d[1.2, 2.5]

    # vector edges can also be forced to discrete mode.
    bo_vec_discrete = BinnedObject(collect(edges), 0.0; interpretation=:discrete)
    pass &= bo_vec_discrete isa BinnedObject{1,Float64,MonteCarloX.DiscreteBinning{Float64}}
    pass &= get_centers(bo_vec_discrete) == collect(edges)

    # invalid interpretation should throw.
    valid = false
    try
        BinnedObject(edges, 0.0; interpretation=:not_a_mode)
    catch err
        valid = err isa ArgumentError
    end
    pass &= valid

    pass &= !hasproperty(bo2d, :weights)

    # return test result
    if verbose
        println("BinnedObject continuous 2D test pass: $(pass)") 
    end

    return pass
end

function run_binned_objects_testsets(; verbose=false)
    @testset "BinnedObject" begin
        @testset "Discrete" begin
            @test test_binned_object_discrete(verbose=verbose)
        end
        @testset "Discrete" begin
            @test test_binned_object_continuous(verbose=verbose)
        end
    end
    return true
end
