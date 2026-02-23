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
    h_local = Histogram((collect(bins),), [1.0, 2.0, 3.0, 4.0])
    # Simulate one rank merging histograms (COMM_SELF)
    pmuca = ParallelMulticanonical(MPI.COMM_SELF, root=0)
    merge_histograms!(pmuca, h_local)
    # Since only one rank, h_local should be unchanged 
    # TODO: how can this be tested with MPI?
    @test all(h_local.weights .== [1.0, 2.0, 3.0, 4.0])

    # Test logweight distribution (broadcast)
    lw = TabulatedLogWeight(bins, 0.0)
    lw.histogram.weights .= [10.0, 20.0, 30.0, 40.0]
    distribute_logweight!(pmuca, lw)
    # Should remain unchanged in COMM_SELF
    @test all(lw.histogram.weights .== [10.0, 20.0, 30.0, 40.0])


    if verbose
        println("ParallelMulticanonical template checks: $(pass)")
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
