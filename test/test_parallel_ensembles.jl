using MonteCarloX
using Random
using StatsBase
using Test
using MPI

function _ensure_mpi_init()
    MPI.Initialized() || MPI.Init()
    return nothing
end

function test_parallel_chains()
    pass = true

    # Threads backend
    alg1 = Metropolis(MersenneTwister(1); β=1.0)
    alg2 = Metropolis(MersenneTwister(2); β=0.5)
    tb = ThreadsBackend(2)
    pc = ParallelChains(tb, [alg1, alg2])
    pass &= check(pc isa ParallelChains{ThreadsBackend}, "ParallelChains Threads type\n")
    pass &= check(rank(pc) == 0, "Threads rank == 0\n")
    pass &= check(size(pc) == 2, "Threads size == 2\n")
    pass &= check(is_root(pc), "Threads is_root\n")
    pass &= check(algorithm(pc, 1) === alg1, "Threads algorithm(1)\n")
    pass &= check(algorithm(pc, 2) === alg2, "Threads algorithm(2)\n")

    # with_parallel
    results = zeros(2)
    with_parallel(pc) do i, alg
        results[i] = ensemble(alg).beta
    end
    pass &= check(results[1] == 1.0, "with_parallel result[1]\n")
    pass &= check(results[2] == 0.5, "with_parallel result[2]\n")

    # merge! with generic collection
    per_chain = [[1.0, 2.0], [3.0, 4.0]]
    merged = merge!(per_chain, +, pc)
    pass &= check(merged == [4.0, 6.0], "merge! Threads\n")

    # size mismatch
    @test_throws ArgumentError ParallelChains(ThreadsBackend(3), [alg1, alg2])

    # MPI backend
    _ensure_mpi_init()
    mb = MPIBackend(MPI.COMM_WORLD)
    alg = Metropolis(MersenneTwister(1); β=1.0)
    pc_mpi = ParallelChains(mb, alg)
    pass &= check(pc_mpi isa ParallelChains{<:MPIBackend}, "ParallelChains MPI type\n")
    pass &= check(rank(pc_mpi) == 0, "MPI rank == 0\n")
    pass &= check(size(pc_mpi) == 1, "MPI size == 1\n")
    pass &= check(is_root(pc_mpi), "MPI is_root\n")
    pass &= check(algorithm(pc_mpi) === alg, "MPI algorithm\n")
    pass &= check(algorithm(pc_mpi, 1) === alg, "MPI algorithm(1)\n")

    # with_parallel MPI
    results_mpi = zeros(1)
    with_parallel(pc_mpi) do alg
        results_mpi[1] = ensemble(alg).beta
    end
    pass &= check(results_mpi[1] == 1.0, "with_parallel MPI result\n")

    # merge! MPI (single rank: unchanged)
    vals = [10.0, 20.0]
    merge!(vals, +, pc_mpi)
    pass &= check(vals == [10.0, 20.0], "merge! MPI unchanged\n")

    return pass
end

function test_parallel_multicanonical()
    _ensure_mpi_init()
    pass = true

    bins = 0.0:1.0:4.0
    muca = Multicanonical(MersenneTwister(1234), BinnedObject(bins, 0.0))

    # MPI backend
    backend = MPIBackend(MPI.COMM_WORLD)
    pmuca = ParallelMulticanonical(backend, muca)
    pass &= check(pmuca isa ParallelChains{<:MPIBackend}, "MPI backend type\n")
    pass &= check(rank(pmuca) == 0, "rank == 0\n")
    pass &= check(size(pmuca) == 1, "size == 1\n")
    pass &= check(is_root(pmuca), "is root\n")

    # merge histograms (single rank: unchanged)
    ensemble(muca).histogram.values .= [1.0, 2.0, 3.0, 4.0]
    merge_histograms!(pmuca)
    pass &= check(all(ensemble(muca).histogram.values .== [1.0, 2.0, 3.0, 4.0]), "merge histograms unchanged\n")

    # distribute logweight (single rank: unchanged)
    ensemble(muca).logweight.values .= [10.0, 20.0, 30.0, 40.0]
    distribute_logweight!(pmuca)
    pass &= check(all(ensemble(muca).logweight.values .== [10.0, 20.0, 30.0, 40.0]), "distribute logweight unchanged\n")

    # Threads backend
    alg1 = Multicanonical(MersenneTwister(1), BinnedObject(bins, 0.0))
    alg2 = Multicanonical(MersenneTwister(2), BinnedObject(bins, 0.0))
    ensemble(alg1).histogram.values .= [1.0, 2.0, 3.0, 4.0]
    ensemble(alg2).histogram.values .= [4.0, 3.0, 2.0, 1.0]
    pmucav = ParallelMulticanonical(ThreadsBackend(2), [alg1, alg2])
    pass &= check(pmucav isa ParallelChains{ThreadsBackend}, "Threads type\n")
    pass &= check(rank(pmucav) == 0, "Threads rank == 0\n")
    pass &= check(size(pmucav) == 2, "Threads size == 2\n")
    pass &= check(is_root(pmucav), "Threads is root\n")

    merge_histograms!(pmucav)
    # merge_histograms! only populates the root chain; other chains are unchanged
    pass &= check(all(ensemble(alg1).histogram.values .== [5.0, 5.0, 5.0, 5.0]), "Threads merge root (alg1)\n")
    pass &= check(all(ensemble(alg2).histogram.values .== [4.0, 3.0, 2.0, 1.0]), "Threads merge non-root unchanged (alg2)\n")

    ensemble(alg1).logweight.values .= [1.0, 2.0, 3.0, 4.0]
    distribute_logweight!(pmucav)
    pass &= check(all(ensemble(alg2).logweight.values .== [1.0, 2.0, 3.0, 4.0]), "Threads distribute logweight\n")

    return pass
end

function test_parallel_tempering()
    _ensure_mpi_init()
    pass = true

    backend = MPIBackend(MPI.COMM_WORLD)
    alg = Metropolis(MersenneTwister(10); β=0.8)
    pt = ParallelTempering(backend, alg)
    pass &= check(rank(pt) == 0, "rank == 0\n")
    pass &= check(size(pt) == 1, "size == 1\n")
    pass &= check(is_root(pt), "is root\n")
    pass &= check(pt isa ReplicaExchange{<:MPIBackend}, "MPI type\n")
    pass &= check(pt.replica.backend === backend, "backend stored\n")
    pass &= check(algorithm(pt) === alg, "alg stored\n")
    pass &= check(index(pt) == 1, "index == 1\n")
    pass &= check(isempty(pt.steps), "steps empty\n")
    pass &= check(isempty(pt.accepted), "accepted empty\n")
    pass &= check(isempty(acceptance_rates(pt)), "acceptance_rates empty\n")
    pass &= check(acceptance_rate(pt) == 0.0, "acceptance_rate == 0.0\n")

    update!(pt, -10.0)
    pass &= check(index(pt) == 1, "index unchanged after update\n")
    pass &= check(ensemble(alg).beta == 0.8, "beta unchanged\n")
    pass &= check(pt.stage == 1, "stage == 1\n")
    pass &= check(isempty(pt.steps), "steps still empty\n")
    pass &= check(isempty(pt.accepted), "accepted still empty\n")

    reset!(pt)
    pass &= check(pt.stage == 0, "stage reset\n")
    pass &= check(index(pt) == 1, "index reset\n")

    # constructor from backend
    rx_backend = ReplicaExchange(backend, alg)
    pass &= check(rx_backend isa ReplicaExchange{<:MPIBackend}, "ReplicaExchange from backend\n")

    # set_betas
    b1 = set_betas(4, 0.4, 1.0, :uniform)
    pass &= check(b1 == [1.0, 0.8, 0.6, 0.4], "uniform betas\n")

    b2 = set_betas(4, 0.5, 1.0, :geometric)
    pass &= check(b2[1] ≈ 1.0, "geometric betas first\n")
    pass &= check(b2[end] ≈ 0.5, "geometric betas last\n")

    # Threads mode
    v_algs = [Metropolis(MersenneTwister(11); β=1.0), Metropolis(MersenneTwister(12); β=0.5)]
    v_pt = ParallelTempering(ThreadsBackend(2), v_algs)
    pass &= check(v_pt isa ReplicaExchange{ThreadsBackend}, "Threads type\n")
    pass &= check(index(v_pt,1) == v_pt.indices[1], "Threads index\n")

    update!(v_pt, [-10.0, -8.0])
    pass &= check(v_pt.stage == 1, "Threads stage == 1\n")
    pass &= check(sum(v_pt.steps) >= 0, "Threads steps >= 0\n")

    # convenience constructor from betas
    v_pt2 = ParallelTempering([1.0, 0.5]; seed=123, rng=MersenneTwister)
    pass &= check(v_pt2 isa ReplicaExchange{ThreadsBackend}, "betas constructor type\n")
    pass &= check(length(v_pt2.replica.alg) == 2, "betas constructor length\n")
    pass &= check(ensemble(algorithm(v_pt2, 1)).beta == 1.0, "betas constructor beta[1]\n")
    pass &= check(ensemble(algorithm(v_pt2, 2)).beta == 0.5, "betas constructor beta[2]\n")

    # acceptance rate helpers
    rates = acceptance_rates(v_pt)
    pass &= check(length(rates) == 1, "acceptance_rates length\n")
    pass &= check(all(0.0 .<= rates .<= 1.0), "acceptance_rates in [0,1]\n")
    pass &= check(0.0 <= acceptance_rate(v_pt) <= 1.0, "acceptance_rate in [0,1]\n")

    # _resolve_pair
    pass &= check(MonteCarloX._resolve_pair(1, 0, 4) == (active=true, pair_id=1, partner_index=2), "resolve_pair (1,0,4)\n")
    pass &= check(MonteCarloX._resolve_pair(2, 0, 4) == (active=true, pair_id=1, partner_index=1), "resolve_pair (2,0,4)\n")
    pass &= check(MonteCarloX._resolve_pair(1, 1, 4) == (active=false, pair_id=0, partner_index=0), "resolve_pair (1,1,4)\n")
    pass &= check(MonteCarloX._resolve_pair(3, 1, 4) == (active=true, pair_id=2, partner_index=2), "resolve_pair (3,1,4)\n")

    # exchange_log_ratio and attempt_exchange_pair!
    alg_i = Metropolis(MersenneTwister(21); β=1.0)
    alg_j = Metropolis(MersenneTwister(22); β=0.5)
    x_i, x_j = 0.0, -5.0
    log_ratio = exchange_log_ratio(ensemble(alg_i), ensemble(alg_j), x_i, x_j)
    pass &= check(isapprox(log_ratio, 2.5; atol=1e-12), "exchange log ratio\n")

    accepted = attempt_exchange_pair!(alg_i, alg_j, x_i, x_j, 0.0)
    pass &= check(accepted, "exchange accepted\n")
    pass &= check(ensemble(alg_i).beta == 0.5, "betas swapped (i)\n")
    pass &= check(ensemble(alg_j).beta == 1.0, "betas swapped (j)\n")

    alg_i_reject = Metropolis(MersenneTwister(23); β=1.0)
    alg_j_reject = Metropolis(MersenneTwister(24); β=0.5)
    rejected = attempt_exchange_pair!(alg_i_reject, alg_j_reject, x_j, x_i, 1.0)
    pass &= check(!rejected, "exchange rejected\n")
    pass &= check(ensemble(alg_i_reject).beta == 1.0, "betas unchanged (i)\n")
    pass &= check(ensemble(alg_j_reject).beta == 0.5, "betas unchanged (j)\n")

    # optimize_exchange_interval!
    v_pt3 = ParallelTempering([1.0, 0.5]; seed=77, rng=MersenneTwister)
    tuple_samples = Tuple{Int,Float64}[
        (1, 0.0), (1, 0.2), (1, -0.1), (1, 0.1), (1, -0.2),
        (2, 0.0), (2, 2.0), (2, -2.0), (2, 2.0), (2, -2.0),
    ]
    sweeps = fill(10, 2)
    interval = optimize_exchange_interval!(
        v_pt3,
        tuple_samples,
        sweeps;
        base_sweeps=10,
        min_sweeps=2,
        max_sweeps=50,
        min_points=2,
        max_lag=2,
    )
    pass &= check(interval == sweeps[index(v_pt3)], "optimize interval consistent\n")
    pass &= check(all(2 .<= sweeps .<= 50), "sweeps in bounds\n")
    pass &= check(sweeps[2] >= sweeps[1], "higher temp needs more sweeps\n")

    return pass
end

@testset "Parallel ensembles" begin
    @testset "Parallel chains" begin
        @test test_parallel_chains()
    end

    @testset "Parallel multicanonical" begin
        @test test_parallel_multicanonical()
    end

    @testset "Parallel tempering" begin
        @test test_parallel_tempering()
    end
end
