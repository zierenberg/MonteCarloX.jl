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
    alg1 = MetropolisAlgorithm(MersenneTwister(1); β=1.0)
    alg2 = MetropolisAlgorithm(MersenneTwister(2); β=0.5)
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
    alg = MetropolisAlgorithm(MersenneTwister(1); β=1.0)
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
    muca = MulticanonicalAlgorithm(MersenneTwister(1234), BinnedObject(bins, 0.0))

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
    alg1 = MulticanonicalAlgorithm(MersenneTwister(1), BinnedObject(bins, 0.0))
    alg2 = MulticanonicalAlgorithm(MersenneTwister(2), BinnedObject(bins, 0.0))
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
    alg = MetropolisAlgorithm(MersenneTwister(10); β=0.8)
    pt = ParallelTempering(backend, alg)
    pass &= check(rank(pt) == 0, "rank == 0\n")
    pass &= check(size(pt) == 1, "size == 1\n")
    pass &= check(is_root(pt), "is root\n")
    pass &= check(pt isa ReplicaExchange{<:MPIBackend}, "MPI type\n")
    pass &= check(pt.replica.backend === backend, "backend stored\n")
    pass &= check(algorithm(pt) === alg, "alg stored\n")
    pass &= check(ensemble_index(pt) == 1, "index == 1\n")
    pass &= check(isempty(pt.steps), "steps empty\n")
    pass &= check(isempty(pt.accepted), "accepted empty\n")
    pass &= check(isempty(acceptance_rates(pt)), "acceptance_rates empty\n")
    pass &= check(acceptance_rate(pt) == 0.0, "acceptance_rate == 0.0\n")

    attempt_exchange!(pt, -10.0)
    pass &= check(ensemble_index(pt) == 1, "index unchanged after update\n")
    pass &= check(ensemble(alg).beta == 0.8, "beta unchanged\n")
    pass &= check(pt.stage == 1, "stage == 1\n")
    pass &= check(isempty(pt.steps), "steps still empty\n")
    pass &= check(isempty(pt.accepted), "accepted still empty\n")

    reset!(pt)
    pass &= check(isempty(pt.steps) && isempty(pt.accepted), "reset clears counters (1 rank)\n")

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
    v_algs = [MetropolisAlgorithm(MersenneTwister(11); β=1.0), MetropolisAlgorithm(MersenneTwister(12); β=0.5)]
    v_pt = ParallelTempering(ThreadsBackend(2), v_algs)
    pass &= check(v_pt isa ReplicaExchange{ThreadsBackend}, "Threads type\n")
    pass &= check(ensemble_index(v_pt,1) == v_pt.indices[1], "Threads index\n")

    attempt_exchange!(v_pt, [-10.0, -8.0])
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
    alg_i = MetropolisAlgorithm(MersenneTwister(21); β=1.0)
    alg_j = MetropolisAlgorithm(MersenneTwister(22); β=0.5)
    x_i, x_j = 0.0, -5.0
    log_ratio = exchange_log_ratio(ensemble(alg_i), ensemble(alg_j), x_i, x_j)
    pass &= check(isapprox(log_ratio, 2.5; atol=1e-12), "exchange log ratio\n")

    accepted = attempt_exchange_pair!(alg_i, alg_j, x_i, x_j, 0.0)
    pass &= check(accepted, "exchange accepted\n")
    pass &= check(ensemble(alg_i).beta == 0.5, "betas swapped (i)\n")
    pass &= check(ensemble(alg_j).beta == 1.0, "betas swapped (j)\n")

    alg_i_reject = MetropolisAlgorithm(MersenneTwister(23); β=1.0)
    alg_j_reject = MetropolisAlgorithm(MersenneTwister(24); β=0.5)
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
    pass &= check(interval == sweeps[ensemble_index(v_pt3)], "optimize interval consistent\n")
    pass &= check(all(2 .<= sweeps .<= 50), "sweeps in bounds\n")
    pass &= check(sweeps[2] >= sweeps[1], "higher temp needs more sweeps\n")

    return pass
end

# Threads replica-exchange *dynamics*: single-process, so the swap loop actually runs (the MPI
# tests can only build a 1-rank backend). Checks a forced swap and the permutation invariant.
function test_replica_exchange_threads_dynamics()
    pass = true
    betas = [1.0, 0.75, 0.5, 0.25]                       # descending β-ladder, 4 replicas
    algs  = [MetropolisAlgorithm(MersenneTwister(300 + i); β = betas[i]) for i in 1:4]
    pt    = ParallelTempering(ThreadsBackend(4), algs)

    # forced accept on edge (1,2): (β₁−β₂)(x₁−x₂) > 0 when the cold replica holds the higher energy
    attempt_exchange!(pt, [10.0, 0.0, -1.0, -2.0])
    pass &= check(sort(pt.indices) == collect(1:4), "ladder stays a permutation\n")
    pass &= check(pt.indices[1] == 2 && pt.indices[2] == 1, "edge (1,2) swapped\n")
    pass &= check(pt.accepted[1] == 1 && pt.steps[1] == 1, "edge-1 attempt+accept counted\n")
    pass &= check(pt.stage == 1, "stage flipped to 1\n")

    # reset! clears statistics only — the ladder permutation and stage are live state
    ladder, stage = copy(pt.indices), pt.stage
    reset!(pt)
    pass &= check(all(pt.steps .== 0) && all(pt.accepted .== 0), "reset clears counters\n")
    pass &= check(pt.indices == ladder && pt.stage == stage, "reset leaves the ladder alone\n")

    # many alternating sweeps: permutation invariant + stage parity hold every step
    pt = ParallelTempering(ThreadsBackend(4),
                           [MetropolisAlgorithm(MersenneTwister(300 + i); β = betas[i]) for i in 1:4])
    rng = MersenneTwister(5)
    for s in 1:50
        attempt_exchange!(pt, randn(rng, 4))
        pass &= check(sort(pt.indices) == collect(1:4), "permutation at sweep $s\n")
        pass &= check(pt.stage == (isodd(s) ? 1 : 0), "stage parity at sweep $s\n")
    end
    pass &= check(all(pt.steps .> 0), "every ladder edge attempted\n")
    rates = acceptance_rates(pt)
    pass &= check(length(rates) == 3 && all(0.0 .<= rates .<= 1.0), "acceptance rates valid\n")
    pass &= check(0.0 <= acceptance_rate(pt) <= 1.0, "overall acceptance rate valid\n")
    return pass
end

# Tempering the strength λ of an auxiliary Hamiltonian term: replica r targets exp(-β(E₀ + λ_r E₁)),
# so the coordinate is the PAIR (E₀, E₁) and the E₀ part must cancel from the exchange ratio by
# itself, leaving β(λᵢ-λⱼ)(E₁ⁱ-E₁ʲ).
const _E0 = [0.0, 1.0, 2.0, 1.5, 0.5, 2.5]
const _E1 = [0.0, -2.0, 3.0, -1.0, 2.0, -3.0]

function test_replica_exchange_composite_coordinate()
    pass = true
    β, λs = 1.0, [1.0, 0.5, 0.0]
    tempered(λ) = FunctionEnsemble(x -> -β * (x[1] + λ * x[2]); linear=true)

    # the shared E₀ part cancels; only the λ-conjugate half survives
    logR = exchange_log_ratio(tempered(1.0), tempered(0.25), (7.0, -2.0), (-3.0, 1.5))
    pass &= check(isapprox(logR, β * (1.0 - 0.25) * (-2.0 - 1.5); atol=1e-12),
                  "composite exchange ratio drops the shared term\n")

    # parameter-schedule constructor: one replica per λ
    rx = ReplicaExchange([tempered(λ) for λ in λs]; seed=9, rng=MersenneTwister)
    pass &= check(logweight(ensemble(algorithm(rx, 1)), (0.0, 1.0)) ≈ -β, "replica 1 carries λ=1\n")
    pass &= check(logweight(ensemble(algorithm(rx, 3)), (0.0, 1.0)) ≈ 0.0, "replica 3 carries λ=0\n")

    # end-to-end: the λ=1 replica must sample exp(-β(E₀+E₁))
    states, counts, nsamples = [Ref(1) for _ in 1:3], zeros(Int, length(_E0)), 4000
    proposal!(s, alg) = begin
        s_new = rand(alg.rng, eachindex(_E0))
        accept!(alg, (_E0[s_new] - _E0[s[]], _E1[s_new] - _E1[s[]])) && (s[] = s_new)
    end
    for _ in 1:nsamples
        advance!(proposal!, rx, states, 20)
        attempt_exchange!(rx, [(_E0[s[]], _E1[s[]]) for s in states])
        for r in eachindex(states)
            ensemble_index(rx, r) == 1 && (counts[states[r][]] += 1)
        end
    end
    exact = (w = exp.(-β .* (_E0 .+ _E1)); w ./ sum(w))
    dev = maximum(abs, counts ./ sum(counts) .- exact)
    pass &= check(dev < 0.03, "λ=1 replica reproduces its target (max dev $dev)\n")
    pass &= check(sort(rx.indices) == collect(1:3), "ladder stays a permutation\n")
    return pass
end

function test_advance_and_ensemble_index()
    pass = true
    betas = [1.0, 0.6, 0.3]
    states = [Ref(-4.0), Ref(-1.0), Ref(2.0)]
    pt = ParallelTempering(betas; seed=5, rng=MersenneTwister)
    attempt_exchange!(pt, [s[] for s in states])

    # the invariant every measurement binning relies on
    for r in eachindex(states)
        pass &= check(ensemble(algorithm(pt, r)).beta == betas[ensemble_index(pt, r)],
                      "replica $r carries the ensemble of its ladder slot\n")
    end

    # advance! must hit every replica — a ReplicaExchange is an AbstractAlgorithm, so it would
    # otherwise fall through to the single-chain method and sample only one
    calls = zeros(Int, 3)
    advance!((s, alg) -> (calls[findfirst(x -> x === s, states)] += 1), pt, states, 4)
    pass &= check(all(calls .== 4), "advance! runs n units on every replica\n")

    serial = 0
    advance!((s, alg) -> (serial += 1), MetropolisAlgorithm(MersenneTwister(1); β=1.0), states[1], 7)
    pass &= check(serial == 7, "single-chain advance! repeats n times\n")
    return pass
end

@testset "Parallel ensembles" begin
    @testset "Parallel chains" begin
        @test test_parallel_chains()
    end

    @testset "Replica exchange dynamics (threads)" begin
        @test test_replica_exchange_threads_dynamics()
    end

    @testset "Parallel multicanonical" begin
        @test test_parallel_multicanonical()
    end

    @testset "Parallel tempering" begin
        @test test_parallel_tempering()
    end

    @testset "Composite reaction coordinate" begin
        @test test_replica_exchange_composite_coordinate()
    end

    @testset "advance! and ensemble_index" begin
        @test test_advance_and_ensemble_index()
    end
end
