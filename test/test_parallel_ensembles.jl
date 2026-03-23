using MonteCarloX
using Random
using StatsBase
using Test
using MPI

function test_parallel_multicanonical(; verbose=false)
    MPI.Init()

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

function run_parallel_ensembles_testsets(; verbose=false)
    @testset "Parallel ensembles" begin
        @testset "Parallel multicanonical" begin
            @test test_parallel_multicanonical(verbose=verbose)
        end

    end
    return true
end
