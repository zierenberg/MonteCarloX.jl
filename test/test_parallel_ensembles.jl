using MonteCarloX
using Random
using StatsBase
using Test
using MPI

function _ensure_mpi_init()
    MPI.Initialized() || MPI.Init()
    return nothing
end

function test_parallel_multicanonical(; verbose=false)
    _ensure_mpi_init()

    pmuca = ParallelMulticanonical(MPI.COMM_WORLD, root=0)

    pass = true
    pass &= pmuca.rank == 0
    pass &= pmuca.size == 1
    pass &= pmuca.root == 0

    bins = 0.0:1.0:4.0
    muca = Multicanonical(MersenneTwister(1234), BinnedObject(bins, 0.0))
    ensemble(muca).histogram.values .= [1.0, 2.0, 3.0, 4.0]
    merge_histograms!(pmuca, ensemble(muca).histogram)
    # Since only one rank, h_local should be unchanged 
    # TODO: how can this be tested with MPI?
    @test all(ensemble(muca).histogram.values .== [1.0, 2.0, 3.0, 4.0])

    # Test logweight distribution (broadcast)
    ensemble(muca).logweight.values .= [10.0, 20.0, 30.0, 40.0]
    distribute_logweight!(pmuca, ensemble(muca).logweight)
    # Should remain unchanged in COMM_SELF
    @test all(ensemble(muca).logweight.values .== [10.0, 20.0, 30.0, 40.0])

    if verbose
        println("ParallelMulticanonical with Multicanonical: $(pass)")
    end

    # implement version also with general histogram of BinnedObject and test that
    hist = Histogram((collect(bins),), zeros(Float64, length(bins) - 1))
    hist.weights .= [1.0, 2.0, 3.0, 4.0]
    merge_histograms!(pmuca, hist)
    @test all(hist.weights .== [1.0, 2.0, 3.0, 4.0])

    lw = BinnedObject(bins, 0.0)
    lw.values .= [10.0, 20.0, 30.0, 40.0]
    distribute_logweight!(pmuca, lw)
    @test all(lw.values .== [10.0, 20.0, 30.0, 40.0])

    if verbose
        println("ParallelMulticanonical with BinnedObject: $(pass)")
    end

    return pass
end

function test_parallel_tempering(; verbose=false)
    _ensure_mpi_init()

    pt = ParallelTempering(MPI.COMM_WORLD, root=0)
    @test pt.rank == 0
    @test pt.size == 1
    @test is_root(pt)
    @test index(pt) == 1
    @test isempty(pt.steps)
    @test isempty(pt.accepted)
    @test isempty(acceptance_rates(pt))
    @test acceptance_rate(pt) == 0.0

    alg = Metropolis(MersenneTwister(10); β=0.8)
    update!(pt, alg, -10.0)
    @test index(pt) == 1
    @test ensemble(alg).beta == 0.8
    @test pt.stage == 1
    @test isempty(pt.steps)
    @test isempty(pt.accepted)

    reset!(pt)
    @test pt.stage == 0
    @test index(pt) == 1

    betas = [1.0, 0.5, 0.2]
    rates = [0.1, 0.6]
    retune_betas!(betas, rates; target=0.3, damping=0.5)
    @test length(betas) == 3
    @test betas[1] > betas[2] > betas[3]

    b1 = set_betas(4, 0.4, 1.0, :uniform)
    @test b1 == [1.0, 0.8, 0.6, 0.4]

    b2 = [1.0, 5/6, 2/3, 0.5]
    set_betas!(b2, [1.0, 0.85, 0.7, 0.5])
    @test b2 == [1.0, 0.85, 0.7, 0.5]

    b3 = set_betas(4, [1.0, 0.9, 0.7, 0.5])
    @test b3 == [1.0, 0.9, 0.7, 0.5]

    b4 = set_betas(4, 0.5, 1.0, :geometric)
    @test b4[1] ≈ 1.0
    @test b4[end] ≈ 0.5

    if verbose
        println("ParallelTempering tests completed")
    end

    return true
end

function run_parallel_ensembles_testsets(; verbose=false)
    @testset "Parallel ensembles" begin
        @testset "Parallel multicanonical" begin
            @test test_parallel_multicanonical(verbose=verbose)
        end

        @testset "Parallel tempering" begin
            @test test_parallel_tempering(verbose=verbose)
        end

    end
    return true
end
