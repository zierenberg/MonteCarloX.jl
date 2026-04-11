using MonteCarloX
using Random
using StatsBase
using Test
using MPI

struct DummyBackend <: AbstractMessageBackend
    myrank::Int
    nranks::Int
end

MonteCarloX.rank(backend::DummyBackend) = backend.myrank
MonteCarloX.size(backend::DummyBackend) = backend.nranks

function _ensure_mpi_init()
    MPI.Initialized() || MPI.Init()
    return nothing
end

function test_parallel_multicanonical()
    _ensure_mpi_init()
    pass = true

    bins = 0.0:1.0:4.0
    muca = Multicanonical(MersenneTwister(1234), BinnedObject(bins, 0.0))

    # MPI message backend
    pmuca = ParallelMulticanonical(MPIBackend(MPI.COMM_WORLD), muca, root=0)
    pass &= check(pmuca isa ParallelMulticanonicalMessage, "MPI backend type\n")
    pass &= check(rank(pmuca) == 0, "rank == 0\n")
    pass &= check(size(pmuca) == 1, "size == 1\n")
    pass &= check(is_root(pmuca), "is root\n")

    # alternative constructor orderings
    pmuca_algfirst = ParallelMulticanonical(muca, MPIBackend(MPI.COMM_WORLD), root=0)
    pass &= check(pmuca_algfirst isa ParallelMulticanonicalMessage, "alg-first constructor\n")

    pmuca_mode = ParallelMulticanonical(muca, :MPI, root=0)
    pass &= check(pmuca_mode isa ParallelMulticanonicalMessage, "mode constructor\n")

    pmuca_nonroot = ParallelMulticanonical(DummyBackend(1, 2), muca; root=0)
    pass &= check(!is_root(pmuca_nonroot), "non-root rank\n")

    # merge histograms (single rank: unchanged)
    ensemble(muca).histogram.values .= [1.0, 2.0, 3.0, 4.0]
    merge_histograms!(pmuca)
    pass &= check(all(ensemble(muca).histogram.values .== [1.0, 2.0, 3.0, 4.0]), "merge histograms unchanged\n")

    # distribute logweight (single rank: unchanged)
    ensemble(muca).logweight.values .= [10.0, 20.0, 30.0, 40.0]
    distribute_logweight!(pmuca)
    pass &= check(all(ensemble(muca).logweight.values .== [10.0, 20.0, 30.0, 40.0]), "distribute logweight unchanged\n")

    # collective operations
    lw = BinnedObject(bins, 0.0)
    lw.values .= [10.0, 20.0, 30.0, 40.0]
    allreduce!(lw.values, +, pmuca)
    pass &= check(all(lw.values .== [10.0, 20.0, 30.0, 40.0]), "allreduce unchanged\n")
    reduced_lw = reduce(lw.values, +, pmuca.root, pmuca)
    pass &= check(reduced_lw == lw.values, "reduce unchanged\n")
    gathered_rank = gather(rank(pmuca), pmuca; root=pmuca.root)
    pass &= check(gathered_rank == [0], "gather rank\n")

    # vector mode
    alg1 = Multicanonical(MersenneTwister(1), BinnedObject(bins, 0.0))
    alg2 = Multicanonical(MersenneTwister(2), BinnedObject(bins, 0.0))
    ensemble(alg1).histogram.values .= [1.0, 2.0, 3.0, 4.0]
    ensemble(alg2).histogram.values .= [4.0, 3.0, 2.0, 1.0]
    pmucav = ParallelMulticanonical([alg1, alg2])
    pass &= check(pmucav isa ParallelMulticanonicalVector, "vector mode type\n")
    pass &= check(rank(pmucav) == 0, "vector rank == 0\n")
    pass &= check(size(pmucav) == 2, "vector size == 2\n")
    pass &= check(is_root(pmucav), "vector is root\n")

    merge_histograms!(pmucav)
    pass &= check(all(ensemble(alg1).histogram.values .== [5.0, 5.0, 5.0, 5.0]), "vector merge alg1\n")
    pass &= check(all(ensemble(alg2).histogram.values .== [5.0, 5.0, 5.0, 5.0]), "vector merge alg2\n")

    ensemble(alg1).logweight.values .= [1.0, 2.0, 3.0, 4.0]
    distribute_logweight!(pmucav)
    pass &= check(all(ensemble(alg2).logweight.values .== [1.0, 2.0, 3.0, 4.0]), "vector distribute logweight\n")

    pass &= check(barrier(pmucav) === nothing, "barrier returns nothing\n")
    pass &= check(allgather(7, pmucav) == [7], "allgather scalar\n")
    v = [3.0, 4.0]
    pass &= check(allreduce!(v, +, pmucav) === v, "allreduce! returns v\n")
    pass &= check(reduce(v, +, 0, pmucav) == [3.0, 4.0], "reduce vector\n")
    pass &= check(MonteCarloX.bcast!(v, 0, pmucav) === v, "bcast! returns v\n")
    pass &= check(gather(9, pmucav; root=0) == [9], "gather scalar\n")

    return pass
end

function test_parallel_tempering()
    _ensure_mpi_init()
    pass = true

    backend = MPIBackend(MPI.COMM_WORLD)
    alg = Metropolis(MersenneTwister(10); β=0.8)
    pt = ParallelTempering(alg, backend; root=0)
    pass &= check(rank(pt) == 0, "rank == 0\n")
    pass &= check(size(pt) == 1, "size == 1\n")
    pass &= check(is_root(pt), "is root\n")
    pass &= check(pt isa ParallelTemperingMessage, "MPI type\n")
    pass &= check(pt.backend === backend, "backend stored\n")
    pass &= check(pt.alg === alg, "alg stored\n")
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

    # alternative constructors
    rx_comm = ReplicaExchange(alg, MPI.COMM_WORLD; root=0)
    pass &= check(rx_comm isa ReplicaExchangeMessage, "ReplicaExchange from comm\n")

    pt_mode = ParallelTempering(alg, :MPI; root=0)
    pass &= check(pt_mode isa ParallelTemperingMessage, "mode constructor\n")

    # retune_betas!
    betas = [1.0, 0.5, 0.2]
    rates = [0.1, 0.6]
    retune_betas!(betas, rates; target=0.3, damping=0.5)
    pass &= check(length(betas) == 3, "betas length unchanged\n")
    pass &= check(betas[1] > betas[2] > betas[3], "betas sorted descending\n")

    # set_betas
    b1 = set_betas(4, 0.4, 1.0, :uniform)
    pass &= check(b1 == [1.0, 0.8, 0.6, 0.4], "uniform betas\n")

    b2 = [1.0, 5/6, 2/3, 0.5]
    set_betas!(b2, [1.0, 0.85, 0.7, 0.5])
    pass &= check(b2 == [1.0, 0.85, 0.7, 0.5], "set_betas! in-place\n")

    b3 = set_betas(4, [1.0, 0.9, 0.7, 0.5])
    pass &= check(b3 == [1.0, 0.9, 0.7, 0.5], "set_betas from vector\n")

    b4 = set_betas(4, 0.5, 1.0, :geometric)
    pass &= check(b4[1] ≈ 1.0, "geometric betas first\n")
    pass &= check(b4[end] ≈ 0.5, "geometric betas last\n")

    # local-vector mode
    v_algs = [Metropolis(MersenneTwister(11); β=1.0), Metropolis(MersenneTwister(12); β=0.5)]
    v_pt = ParallelTempering(v_algs)
    pass &= check(v_pt isa ParallelTemperingVector, "vector type\n")
    pass &= check(index(v_pt, 1) == 1, "vector index\n")

    update!(v_pt, [-10.0, -8.0])
    pass &= check(v_pt.stage == 1, "vector stage == 1\n")
    pass &= check(sum(v_pt.steps) >= 0, "vector steps >= 0\n")
    pass &= check(gather_at_root(5, v_pt) == [5], "gather_at_root\n")
    arr = [1, 2]
    pass &= check(broadcast_from_root!(arr, v_pt) === arr, "broadcast_from_root!\n")

    # convenience constructor from betas
    v_pt2 = ParallelTempering([1.0, 0.5]; seed=123, rng=MersenneTwister)
    pass &= check(v_pt2 isa ParallelTemperingVector, "betas constructor type\n")
    pass &= check(length(v_pt2.alg) == 2, "betas constructor length\n")
    pass &= check(ensemble(v_pt2.alg[1]).beta == 1.0, "betas constructor beta[1]\n")
    pass &= check(ensemble(v_pt2.alg[2]).beta == 0.5, "betas constructor beta[2]\n")

    # acceptance rate helpers
    pass &= check(MonteCarloX.acceptance_rates([4, 0, 3], [1, 0, 3]) == [0.25, 0.0, 1.0], "acceptance_rates vector\n")
    pass &= check(MonteCarloX.acceptance_rate([4, 0, 3], [1, 0, 3]) == 4 / 7, "acceptance_rate scalar\n")

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
    @testset "Parallel multicanonical" begin
        @test test_parallel_multicanonical()
    end

    @testset "Parallel tempering" begin
        @test test_parallel_tempering()
    end
end
