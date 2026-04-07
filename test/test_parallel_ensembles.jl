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
    @test exchange_partner(pt, 0) == -1
    @test exchange_partner(pt, 1) == -1

    out = attempt_exchange!(pt, 0.8, -10.0; rng=MersenneTwister(1), stage=0)
    @test out.accepted == false
    @test out.partner == -1
    @test out.beta == 0.8

    betas = [1.0, 0.5, 0.2]
    energies = [10.0, -10.0, 0.0]
    stats = ExchangeStats(length(betas) - 1)
    attempt_exchange_pairs!(MersenneTwister(2), betas, energies, 0; stats=stats)
    @test betas == [0.5, 1.0, 0.2]
    @test stats.attempts == [1, 0]
    @test stats.accepts == [1, 0]
    @test acceptance_rates(stats) == [1.0, 0.0]

    algs = [Metropolis(MersenneTwister(10 + i); β=β) for (i, β) in enumerate([1.0, 0.5, 0.2])]
    ens1 = ensemble(algs[1])
    ens2 = ensemble(algs[2])
    labels = [1, 2, 3]
    stats2 = ExchangeStats(2)
    attempt_exchange_pairs!(MersenneTwister(2), algs, energies, 0; stats=stats2, labels=labels)
    @test ensemble(algs[1]) === ens2
    @test ensemble(algs[2]) === ens1
    @test inverse_temperature(algs[1]) == 0.5
    @test inverse_temperature(algs[2]) == 1.0
    @test labels == [2, 1, 3]
    @test stats2.attempts == [1, 0]
    @test stats2.accepts == [1, 0]

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
