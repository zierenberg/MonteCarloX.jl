using MonteCarloX
using StatsBase
using Test

function test_binned_object_discrete()
    pass = true

    # 1D
    bins = 0:2:10
    bo = BinnedObject(bins)
    pass &= check(bo isa BinnedObject{1,Float64,DiscreteBinning{Int64}}, "1D discrete type\n")
    pass &= check(all(iszero, bo.values), "values zeroed\n")
    pass &= check(size(bo) == (6,), "size == (6,)\n")
    pass &= check(bo.bins[1].start == 0, "start == 0\n")
    pass &= check(bo.bins[1].step == 2, "step == 2\n")
    pass &= check(get_centers(bo) == [0, 2, 4, 6, 8, 10], "centers correct\n")

    bo[4] = 1.5
    pass &= check(bo[4] == 1.5, "getindex after setindex\n")
    pass &= check(bo(4) == bo[4], "callable == getindex\n")

    # vector domain constructor
    bins_vec = [0, 2, 4, 6, 8, 10]
    bo_vec = BinnedObject(bins_vec, 0.0)
    pass &= check(bo_vec isa BinnedObject{1,Float64,DiscreteBinning{Int64}}, "vector domain type\n")
    pass &= check(all(iszero, bo_vec.values), "vector domain zeroed\n")
    pass &= check(size(bo_vec) == (6,), "vector domain size\n")
    pass &= check(bo_vec.bins[1].start == 0, "vector domain start\n")
    pass &= check(bo_vec.bins[1].step == 2, "vector domain step\n")
    pass &= check(get_centers(bo_vec) == bins_vec, "vector domain centers\n")
    pass &= check(get_edges(bo_vec) == -1:2:11, "infer right edges\n")

    # single-bin vector throws
    threw = try; BinnedObject([0], 0.0); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "single-bin throws\n")

    # non-equidistant bins throw
    threw = try; BinnedObject([0, 1, 3], 0.0); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "non-equidistant throws\n")

    # unsupported domain type throws
    threw = try; BinnedObject("invalid domain", 0.0); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "unsupported domain throws\n")

    # non-integer DiscreteBinning
    bins_float = 0.0:2.0:10.0
    b = DiscreteBinning(first(bins_float), step(bins_float), length(bins_float))
    pass &= check(b isa DiscreteBinning{Float64}, "float DiscreteBinning type\n")
    pass &= check(get_centers(b) == collect(bins_float), "float DiscreteBinning centers\n")
    pass &= check(get_edges(b) == collect(-1.0:2.0:11.0), "float DiscreteBinning edges\n")
    pass &= check(MonteCarloX._binindex(b, 4.0) == 3, "float DiscreteBinning binindex\n")

    # _assert_same_domain
    bo2 = BinnedObject(0:2:10, 0.0)
    pass &= check(MonteCarloX._assert_same_domain(bo, bo2) == nothing, "same domain passes\n")
    bo3 = BinnedObject(0:1:10, 0.0)
    threw = try; MonteCarloX._assert_same_domain(bo, bo3); false; catch err; err isa AssertionError; end
    pass &= check(threw, "different domain throws\n")

    # 2D
    bins2d = (0:1:5, 0:2:10)
    bo2d = BinnedObject(bins2d, 0.0)
    pass &= check(bo2d isa BinnedObject{2, Float64, DiscreteBinning{Int64}}, "2D discrete type\n")
    pass &= check(all(iszero, bo2d.values), "2D values zeroed\n")
    pass &= check(size(bo2d) == (6, 6), "2D size\n")
    pass &= check(bo2d.bins[1].start == 0, "2D dim1 start\n")
    pass &= check(bo2d.bins[2].start == 0, "2D dim2 start\n")
    pass &= check(bo2d.bins[1].step == 1, "2D dim1 step\n")
    pass &= check(bo2d.bins[2].step == 2, "2D dim2 step\n")

    bo2d[3, 4] = -0.5
    pass &= check(bo2d[3, 4] == -0.5, "2D getindex\n")
    pass &= check(bo2d(3, 4) == bo2d[3, 4], "2D callable\n")

    return pass
end

function test_binned_object_continuous()
    pass = true

    # 1D
    range = 0.0:1.0:4.0
    bo = BinnedObject(range, 0.0)
    pass &= check(bo isa BinnedObject{1,Float64,ContinuousBinning{Float64}}, "1D continuous type\n")
    pass &= check(all(iszero, bo.values), "values zeroed\n")
    pass &= check(size(bo) == (4,), "size == (4,)\n")
    pass &= check(bo.bins[1].edges == collect(range), "edges match range\n")
    pass &= check(bo.bins[1].centers == [0.5, 1.5, 2.5, 3.5], "centers correct\n")

    # continuous interpretation from integer vector
    range_int = 0:1:4
    bo_vec = BinnedObject(collect(range_int), 0.0; interpretation=:continuous)
    pass &= check(bo_vec isa BinnedObject{1,Float64,ContinuousBinning{Float64}}, "vector continuous type\n")
    pass &= check(bo_vec == bo, "vector continuous matches range\n")

    # explicit discrete mode for float ranges
    bo_discrete_float = BinnedObject(range, 0.0; interpretation=:discrete)
    pass &= check(bo_discrete_float isa BinnedObject{1,Float64,DiscreteBinning{Float64}}, "discrete float type\n")
    pass &= check(get_centers(bo_discrete_float) == collect(range), "discrete float centers\n")
    pass &= check(get_edges(bo_discrete_float) == collect(-0.5:1.0:4.5), "discrete float edges\n")

    # indexing by coordinate
    bo[1.2] = 1.5
    pass &= check(bo.values[2] == 1.5, "setindex! by coordinate\n")
    pass &= check(bo[1.2] == 1.5, "getindex by coordinate\n")
    pass &= check(bo(1.2) == bo[1.2], "callable == getindex\n")

    # _assert_same_domain
    bo2 = BinnedObject(range, 0.0)
    pass &= check(MonteCarloX._assert_same_domain(bo, bo2) == nothing, "same domain passes\n")

    bo3 = BinnedObject(0.0:0.5:4.0, 0.0)
    threw = try; MonteCarloX._assert_same_domain(bo, bo3); false; catch err; err isa AssertionError; end
    pass &= check(threw, "different continuous domain throws\n")

    bo_discrete = BinnedObject(0:1:4, 0.0)
    threw = try; MonteCarloX._assert_same_domain(bo, bo_discrete); false; catch err; err isa AssertionError; end
    pass &= check(threw, "continuous vs discrete throws\n")

    # 2D
    range2d = (0.0:1.0:3.0, 0.0:2.0:6.0)
    bo2d = BinnedObject(range2d, 0.0)
    pass &= check(bo2d isa BinnedObject{2,Float64,ContinuousBinning{Float64}}, "2D continuous type\n")
    pass &= check(all(iszero, bo2d.values), "2D values zeroed\n")
    pass &= check(size(bo2d) == (3, 3), "2D size\n")
    pass &= check(bo2d.bins[1].edges == collect(range2d[1]), "2D dim1 edges match\n")
    pass &= check(bo2d.bins[2].edges == collect(range2d[2]), "2D dim2 edges match\n")
    pass &= check(bo2d.bins[1].centers == [0.5, 1.5, 2.5], "2D dim1 centers\n")
    pass &= check(bo2d.bins[2].centers == [1.0, 3.0, 5.0], "2D dim2 centers\n")

    bo2d[1.2, 2.5] -= 0.5
    pass &= check(bo2d.values[2,2] == -0.5, "2D setindex!\n")
    pass &= check(bo2d[1.2, 2.5] == -0.5, "2D getindex\n")
    pass &= check(bo2d(1.2, 2.5) == bo2d[1.2, 2.5], "2D callable\n")

    # vector edges forced to discrete
    bo_vec_discrete = BinnedObject(collect(range), 0.0; interpretation=:discrete)
    pass &= check(bo_vec_discrete isa BinnedObject{1,Float64,DiscreteBinning{Float64}}, "vector discrete type\n")
    pass &= check(get_centers(bo_vec_discrete) == collect(range), "vector discrete centers\n")

    # invalid interpretation throws
    threw = try; BinnedObject(range, 0.0; interpretation=:not_a_mode); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "invalid interpretation throws\n")

    return pass
end

@testset "BinnedObject" begin
    @testset "Discrete" begin
        @test test_binned_object_discrete()
    end
    @testset "Continuous" begin
        @test test_binned_object_continuous()
    end
end
