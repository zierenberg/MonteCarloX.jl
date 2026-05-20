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
    pass &= check(get_edges(bo) == collect(range), "edges match range\n")
    pass &= check(get_centers(bo) == [0.5, 1.5, 2.5, 3.5], "centers correct\n")

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
    pass &= check(get_edges(bo2d, 1) == collect(range2d[1]), "2D dim1 edges match\n")
    pass &= check(get_edges(bo2d, 2) == collect(range2d[2]), "2D dim2 edges match\n")
    pass &= check(get_centers(bo2d, 1) == [0.5, 1.5, 2.5], "2D dim1 centers\n")
    pass &= check(get_centers(bo2d, 2) == [1.0, 3.0, 5.0], "2D dim2 centers\n")

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

function test_binned_object_arbitrary_continuous()
    pass = true

    # ArbitraryContinuousBinning from non-uniform vector
    edges = [0.0, 1.0, 3.0, 6.0, 10.0]
    bo = BinnedObject(edges, 0.0; interpretation=:continuous)
    pass &= check(bo isa BinnedObject{1,Float64,ArbitraryContinuousBinning{Float64}}, "arbitrary continuous type\n")
    pass &= check(size(bo) == (4,), "size == (4,)\n")
    pass &= check(get_edges(bo) == edges, "edges preserved\n")
    pass &= check(get_centers(bo) == [0.5, 2.0, 4.5, 8.0], "centers correct\n")

    # indexing
    bo[0.5] = 2.0
    pass &= check(bo.values[1] == 2.0, "setindex! first bin\n")
    bo[5.0] = 3.0
    pass &= check(bo.values[3] == 3.0, "setindex! middle bin\n")
    pass &= check(bo(5.0) == 3.0, "callable matches\n")

    # promotion from ContinuousBinning
    uniform = ContinuousBinning(0.0, 1.0, 4)
    arb = ArbitraryContinuousBinning(uniform)
    pass &= check(arb isa ArbitraryContinuousBinning{Float64}, "promotion type\n")
    pass &= check(get_edges(arb) == get_edges(uniform), "promotion edges match\n")
    pass &= check(get_centers(arb) == get_centers(uniform), "promotion centers match\n")

    return pass
end

function test_binned_object_boundary()
    pass = true

    # ── ErrorBoundary (default) ──
    bo_err = BinnedObject(0:4, 1.0)
    threw = try; bo_err(-1); false; catch; true; end
    pass &= check(threw, "ErrorBoundary read throws on OOB\n")
    threw = try; bo_err[-1] = 5.0; false; catch; true; end
    pass &= check(threw, "ErrorBoundary write throws on OOB\n")

    # ── NegInfBoundary ──
    bo_lw = BinnedObject(0:4, 1.0; boundary=NegInfBoundary())
    pass &= check(bo_lw(-1) == -Inf, "NegInfBoundary returns -Inf below\n")
    pass &= check(bo_lw(5) == -Inf, "NegInfBoundary returns -Inf above\n")
    pass &= check(bo_lw(2) == 1.0, "NegInfBoundary returns value in-bounds\n")
    # setindex! is no-op for OOB
    bo_lw[-1] = 99.0
    pass &= check(all(bo_lw.values .== 1.0), "NegInfBoundary setindex! no-op OOB\n")

    # ── ZeroBoundary ──
    bo_hist = BinnedObject(0:4, 1.0; boundary=ZeroBoundary())
    pass &= check(bo_hist(-1) == 0.0, "ZeroBoundary returns zero below\n")
    pass &= check(bo_hist(5) == 0.0, "ZeroBoundary returns zero above\n")
    pass &= check(bo_hist(2) == 1.0, "ZeroBoundary returns value in-bounds\n")
    # setindex! is no-op for OOB
    bo_hist[-1] = 99.0
    pass &= check(all(bo_hist.values .== 1.0), "ZeroBoundary setindex! no-op OOB\n")
    # += pattern safe for OOB
    bo_hist[-1] += 1
    pass &= check(all(bo_hist.values .== 1.0), "ZeroBoundary += no-op OOB\n")

    # ── Boundary preserved by zero() ──
    z = zero(bo_lw)
    pass &= check(z(-1) == -Inf, "zero() preserves NegInfBoundary\n")

    # ── Continuous with boundary ──
    bo_cont = BinnedObject(0.0:1.0:4.0, 0.0; boundary=NegInfBoundary())
    pass &= check(bo_cont(-0.5) == -Inf, "continuous NegInfBoundary below\n")
    pass &= check(bo_cont(4.5) == -Inf, "continuous NegInfBoundary above\n")
    pass &= check(bo_cont(0.5) == 0.0, "continuous NegInfBoundary in-bounds\n")

    # ── 2D with boundary ──
    bo2d = BinnedObject((0:2, 0:2), 1.0; boundary=ZeroBoundary())
    pass &= check(bo2d(-1, 1) == 0.0, "2D ZeroBoundary OOB dim1\n")
    pass &= check(bo2d(1, -1) == 0.0, "2D ZeroBoundary OOB dim2\n")
    pass &= check(bo2d(1, 1) == 1.0, "2D ZeroBoundary in-bounds\n")

    return pass
end

function test_binned_object_interpolation()
    pass = true

    # Source: coarse grid with known linear values  f(x) = 2x
    source = BinnedObject(0:2:10, 0.0)
    for (i, x) in enumerate(get_centers(source))
        source.values[i] = 2.0 * x
    end

    # Target: finer grid, same domain
    target = BinnedObject(0:1:10, 0.0)
    set!(target, source)
    for (i, x) in enumerate(get_centers(target))
        pass &= check(target.values[i] ≈ 2.0 * x, "interpolation at x=$x\n")
    end

    # Target extends beyond source domain — constant extrapolation
    target_wide = BinnedObject(-2:1:12, 0.0)
    set!(target_wide, source)
    pass &= check(target_wide.values[1] ≈ source.values[1], "extrapolation below\n")
    pass &= check(target_wide.values[end] ≈ source.values[end], "extrapolation above\n")

    # Continuous binning
    source_cont = BinnedObject(0.0:1.0:4.0, 0.0)  # edges, 4 bins with centers 0.5,1.5,2.5,3.5
    src_cs = get_centers(source_cont)
    for (i, x) in enumerate(src_cs)
        source_cont.values[i] = x^2
    end
    target_cont = BinnedObject(0.0:0.5:4.0, 0.0)  # finer edges, centers 0.25,0.75,1.25,...
    set!(target_cont, source_cont)
    # target center 0.75 is quarter-way between src centers 0.5 and 1.5
    # src values: 0.25 (at 0.5), 2.25 (at 1.5)
    # t = (0.75 - 0.5) / (1.5 - 0.5) = 0.25, expect 0.75*0.25 + 0.25*2.25 = 0.75
    tgt_cs = get_centers(target_cont)
    idx = findfirst(x -> x ≈ 0.75, tgt_cs)
    pass &= check(target_cont.values[idx] ≈ 0.75, "continuous interpolation\n")

    # rescale_bins: source in absolute coords, target in rescaled coords
    # source bins at E = 0,10,20 with values 0,1,2; rescale_bins=2 maps to 0,20,40
    source_abs = BinnedObject(0:10:20, 0.0)
    source_abs.values .= [0.0, 1.0, 2.0]
    target_rescaled = BinnedObject(0:5:40, 0.0)
    set!(target_rescaled, source_abs; rescale_bins=2)
    # target center 10 maps to source x=10/2=5, midpoint of src 0 and 10 → 0.5
    pass &= check(target_rescaled[10] ≈ 0.5, "rescale_bins interpolation\n")
    # target center 20 maps to source x=20/2=10 → 1.0
    pass &= check(target_rescaled[20] ≈ 1.0, "rescale_bins exact match\n")

    # rescale_values
    set!(target_rescaled, source_abs; rescale_bins=2, rescale_values=3.0)
    pass &= check(target_rescaled[20] ≈ 3.0, "rescale_values\n")

    # Source with too few bins throws
    source_tiny = BinnedObject(0:0, 0.0)
    threw = try; set!(target, source_tiny); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "single-bin source throws\n")

    return pass
end

@testset "BinnedObject" begin
    @testset "Discrete" begin
        @test test_binned_object_discrete()
    end
    @testset "Continuous" begin
        @test test_binned_object_continuous()
    end
    @testset "ArbitraryContinuous" begin
        @test test_binned_object_arbitrary_continuous()
    end
    @testset "Boundary" begin
        @test test_binned_object_boundary()
    end
    @testset "Interpolation" begin
        @test test_binned_object_interpolation()
    end
end
