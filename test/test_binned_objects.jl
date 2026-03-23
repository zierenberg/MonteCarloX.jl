using MonteCarloX
using StatsBase
using Test

# test for BinnedObject
function test_binned_object_discrete(; verbose=false)
    pass = true 

    # 1D
    bins = 0:2:10
    bo = BinnedObject(bins) # default constructor with init=0.0
    pass &= bo isa BinnedObject{1,Float64,DiscreteBinning{Int64}}
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
    pass &= bo_vec isa BinnedObject{1,Float64,DiscreteBinning{Int64}}
    pass &= all(iszero, bo_vec.values)
    pass &= size(bo_vec) == (6,)
    pass &= size(bo_vec.values) == (6,)
    pass &= bo_vec.bins[1].start == 0
    pass &= bo_vec.bins[1].step == 2
    pass &= get_centers(bo_vec) == bins_vec

    # for discrete bins, edges indeed defined so that bins are at their centers
    edges_vec = -1:2:11
    pass &= get_edges(bo_vec) == edges_vec

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
    b = DiscreteBinning(first(bins_float), step(bins_float), length(bins_float))
    pass &= b isa DiscreteBinning{Float64}
    pass &= get_centers(b) == collect(bins_float)
    pass &= get_edges(b) == collect(-1.0:2.0:11.0)
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
    pass &= bo2d isa BinnedObject{2, Float64, DiscreteBinning{Int64}}
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

function (test_binned_object_continuous)(; verbose=false)
    pass = true 

    # 1D
    range = 0.0:1.0:4.0
    bo = BinnedObject(range, 0.0)
    local ok
    ok = bo isa BinnedObject{1,Float64,ContinuousBinning{Float64}}
        if verbose println(ok, " isa") end
    pass &= ok
    ok = all(iszero, bo.values)
        if verbose println(ok, " all iszero, values=", bo.values) end
    pass &= ok
    ok = size(bo) == (4,)
        if verbose println(ok, " size, got=", size(bo), " expected=(4,)") end
    pass &= ok
    ok = size(bo.values) == (4,)
        if verbose println(ok, " size(values), got=", size(bo.values), " expected=(4,)") end
    pass &= ok
    ok = length(bo.bins[1].edges) == 5
        if verbose println(ok, " length(edges), got=", length(bo.bins[1].edges), " expected=5") end
    pass &= ok
    ok = length(bo.bins[1].centers) == 4
        if verbose println(ok, " length(centers), got=", length(bo.bins[1].centers), " expected=4") end
    pass &= ok
    ok = bo.bins[1].edges == collect(range)
        if verbose println(ok, " edges==range, got=", bo.bins[1].edges, " expected=", collect(range)) end
    pass &= ok
    ok = bo.bins[1].centers == [0.5, 1.5, 2.5, 3.5]
        if verbose println(ok, " centers==[0.5,1.5,2.5,3.5], got=", bo.bins[1].centers) end
    pass &= ok

    # get the same with the continuous interpretation even if the input is a vector of floats.
    range_int = 0:1:4
    bo_vec = BinnedObject(collect(range_int), 0.0; interpretation=:continuous)
    ok = bo_vec isa BinnedObject{1,Float64,ContinuousBinning{Float64}}
        if verbose println(ok, " bo_vec isa ContinuousBinning") end
    pass &= ok
    ok = bo_vec == bo
        if verbose println(ok, " bo_vec == bo, bo_vec=", bo_vec, " bo=", bo) end
    pass &= ok

    # float ranges are interpreted as continuous edges by default;
    # explicit mode can force discrete center interpretation.
    bo_discrete_float = BinnedObject(range, 0.0; interpretation=:discrete)
    pass &= bo_discrete_float isa BinnedObject{1,Float64,DiscreteBinning{Float64}}
    pass &= get_centers(bo_discrete_float) == collect(range)
    pass &= get_edges(bo_discrete_float) == collect(-0.5:1.0:4.5)

    bo[1.2] = 1.5
    ok = bo.values[2] == 1.5
        if verbose println(ok, " bo.values[2] == 1.5, got=", bo.values[2]) end
    pass &= ok
    ok = bo[1.2] == 1.5
        if verbose println(ok, " bo[1.2] == 1.5, got=", bo[1.2]) end
    pass &= ok
    ok = bo(1.2) == bo[1.2]
        if verbose println(ok, " bo(1.2) == bo[1.2], got=", bo(1.2), " expected=", bo[1.2]) end
    pass &= ok

    ok = !hasproperty(bo, :weights)
        if verbose println(ok, " !hasproperty(bo, :weights)") end
    pass &= ok

    binvals = get_centers(bo)
    ok = length(binvals) == length(range) - 1
        if verbose println(ok, " length(binvals) == length(range)-1, got=", length(binvals), " expected=", length(range)-1) end
    pass &= ok
    ok = get_edges(bo) == collect(range)
        if verbose println(ok, " get_edges(bo) == range, got=", get_edges(bo), " expected=", collect(range)) end
    pass &= ok
    ok = get_centers(bo) == [0.5, 1.5, 2.5, 3.5]
        if verbose println(ok, " get_centers(bo) == [0.5,1.5,2.5,3.5], got=", get_centers(bo)) end
    pass &= ok

    # test _assert_same_domain
    bo2 = BinnedObject(range, 0.0)
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
        @show bo
        @show bo.values
        @show bo.bins[1].edges
        @show bo.bins[1].centers
        @show get_centers(bo)
        @show get_edges(bo)
    end

    if verbose
        println("BinnedObject continuous 1D test pass: $(pass)")
    end

    # 2D
    range2d = (0.0:1.0:3.0, 0.0:2.0:6.0)
    bo2d = BinnedObject(range2d, 0.0)
    pass &= bo2d isa BinnedObject{2,Float64,ContinuousBinning{Float64}}
    pass &= all(iszero, bo2d.values)
    pass &= size(bo2d) == (3, 3)
    pass &= size(bo2d.values) == (3, 3)
    pass &= length(bo2d.bins[1].edges) == 4
    pass &= length(bo2d.bins[2].edges) == 4
    pass &= length(bo2d.bins[1].centers) == 3
    pass &= length(bo2d.bins[2].centers) == 3
    pass &= bo2d.bins[1].edges == collect(range2d[1])
    pass &= bo2d.bins[2].edges == collect(range2d[2])
    pass &= bo2d.bins[1].centers == [0.5, 1.5, 2.5]
    pass &= bo2d.bins[2].centers == [1.0, 3.0, 5.0]

    bo2d[1.2, 2.5] -= 0.5
    pass &= bo2d.values[2,2] == -0.5 
    pass &= bo2d[1.2, 2.5] == -0.5
    pass &= bo2d(1.2, 2.5) == bo2d[1.2, 2.5]

    # vector edges can also be forced to discrete mode.
    bo_vec_discrete = BinnedObject(collect(range), 0.0; interpretation=:discrete)
    pass &= bo_vec_discrete isa BinnedObject{1,Float64,DiscreteBinning{Float64}}
    pass &= get_centers(bo_vec_discrete) == collect(range)

    # invalid interpretation should throw.
    valid = false
    try
        BinnedObject(range, 0.0; interpretation=:not_a_mode)
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
        @testset "Continuous" begin
            @test test_binned_object_continuous(verbose=verbose)
        end
    end
    return true
end
