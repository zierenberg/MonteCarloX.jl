using MonteCarloX
using Random
using StatsBase
using Test

function test_replica_exchange_templates(; verbose=false)
    rng = MersenneTwister(1301)
    μ = [-2.0, 0.0, 2.0, 4.0]

    replicas = [Metropolis(MersenneTwister(500 + i), x -> -0.5 * (x - μ[i])^2) for i in eachindex(μ)]
    re = ReplicaExchange(replicas; rng=rng, rank=0, comm_size=2, local_indices=[1, 2])

    states = [-3.0, 1.0, 2.5, 3.5]

    pass = true
    pass &= length(re.parallel.replicas) == 4
    pass &= length(re.parallel.local_indices) == 2

    attempts0 = re.exchange_attempts
    _ = attempt_exchange!(re, states, 1, 2)
    pass &= re.exchange_attempts == attempts0 + 1
    pass &= 0.0 <= exchange_rate(re) <= 1.0

    reset_exchange_statistics!(re)
    pass &= re.exchange_attempts == 0
    pass &= re.exchange_accepted == 0

    if verbose
        println("ReplicaExchange template checks: $(pass)")
    end

    return pass
end

function test_parallel_tempering_templates(; verbose=false)
    rng = MersenneTwister(1201)

    replicas = [
        Metropolis(MersenneTwister(1); β=0.5),
        Metropolis(MersenneTwister(2); β=1.0),
        Metropolis(MersenneTwister(3); β=1.5),
        Metropolis(MersenneTwister(4); β=2.0),
    ]

    pt = ParallelTempering(replicas; rng=rng, rank=0, comm_size=2, local_indices=[1, 2])

    pass = true
    pass &= length(pt.parallel.replicas) == 4
    pass &= length(pt.parallel.local_indices) == 2
    pass &= pt.parallel.rank == 0
    pass &= pt.parallel.comm_size == 2

    energies = [0.0, 10.0, 20.0, 30.0]

    accepted = attempt_exchange!(pt, energies, 1, 2)
    pass &= accepted
    pass &= energies[1] == 10.0 && energies[2] == 0.0
    pass &= pt.exchange_attempts == 1
    pass &= pt.exchange_accepted == 1
    pass &= exchange_rate(pt) == 1.0

    reset_exchange_statistics!(pt)
    pass &= pt.exchange_attempts == 0
    pass &= pt.exchange_accepted == 0

    if verbose
        println("ParallelTempering template checks: $(pass)")
    end

    return pass
end

function test_parallel_multicanonical_templates(; verbose=false)
    bins = 0.0:1.0:4.0

    lw1 = TabulatedLogWeight(Histogram((collect(bins),), zeros(Float64, length(bins) - 1)))
    lw2 = TabulatedLogWeight(Histogram((collect(bins),), zeros(Float64, length(bins) - 1)))

    r1 = Multicanonical(MersenneTwister(11), lw1)
    r2 = Multicanonical(MersenneTwister(12), lw2)

    pmuca = ParallelMulticanonical([r1, r2]; rank=1, comm_size=4, master_rank=0)

    pass = true
    pass &= length(pmuca.parallel.replicas) == 2
    pass &= length(pmuca.parallel.local_indices) == 2
    pass &= pmuca.parallel.rank == 1
    pass &= pmuca.parallel.comm_size == 4
    pass &= pmuca.master_rank == 0

    h1 = fit(Histogram, [0.2, 0.8, 1.2], bins)
    h2 = fit(Histogram, [2.2, 2.5, 2.8], bins)

    pass &= update_weights!(pmuca, [h1, h2]; mode=:simple, master_replica_index=1) === nothing

    pass &= all(isapprox.(lw1.table.weights, lw2.table.weights))
    pass &= any(!iszero, lw1.table.weights)

    if verbose
        println("ParallelMulticanonical template checks: $(pass)")
    end

    return pass
end

function run_parallel_ensembles_testsets(; verbose=false)
    @testset "Parallel ensembles" begin
        @testset "Replica exchange template" begin
            @test test_replica_exchange_templates(verbose=verbose)
        end
        @testset "Parallel tempering template" begin
            @test test_parallel_tempering_templates(verbose=verbose)
        end
        @testset "Parallel multicanonical template" begin
            @test test_parallel_multicanonical_templates(verbose=verbose)
        end
    end
    return true
end
