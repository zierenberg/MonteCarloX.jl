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

function test_parallel_multicanonical(; verbose=false)
    _ensure_mpi_init()

    bins = 0.0:1.0:4.0
    muca = Multicanonical(MersenneTwister(1234), BinnedObject(bins, 0.0))

    pmuca = ParallelMulticanonical(MPIBackend(MPI.COMM_WORLD), muca, root=0)
    @test pmuca isa ParallelMulticanonicalMessage
    @test rank(pmuca) == 0
    @test size(pmuca) == 1
    @test is_root(pmuca)

    pmuca_nonroot = ParallelMulticanonical(DummyBackend(1, 2), muca; root=0)
    @test !is_root(pmuca_nonroot)

    ensemble(muca).histogram.values .= [1.0, 2.0, 3.0, 4.0]
    merge_histograms!(pmuca)
    # Since only one rank, histogram should be unchanged
    @test all(ensemble(muca).histogram.values .== [1.0, 2.0, 3.0, 4.0])

    # Test logweight distribution (broadcast)
    ensemble(muca).logweight.values .= [10.0, 20.0, 30.0, 40.0]
    distribute_logweight!(pmuca)
    # Should remain unchanged with a single-rank communicator
    @test all(ensemble(muca).logweight.values .== [10.0, 20.0, 30.0, 40.0])

    lw = BinnedObject(bins, 0.0)
    lw.values .= [10.0, 20.0, 30.0, 40.0]
    allreduce!(lw.values, +, pmuca)
    @test all(lw.values .== [10.0, 20.0, 30.0, 40.0])
    reduced_lw = reduce(lw.values, +, pmuca.root, pmuca)
    @test reduced_lw == lw.values
    gathered_rank = gather(rank(pmuca), pmuca; root=pmuca.root)
    @test gathered_rank == [0]

    # Test vector mode
    alg1 = Multicanonical(MersenneTwister(1), BinnedObject(bins, 0.0))
    alg2 = Multicanonical(MersenneTwister(2), BinnedObject(bins, 0.0))
    ensemble(alg1).histogram.values .= [1.0, 2.0, 3.0, 4.0]
    ensemble(alg2).histogram.values .= [4.0, 3.0, 2.0, 1.0]
    pmucav = ParallelMulticanonical([alg1, alg2])
    @test pmucav isa ParallelMulticanonicalVector
    @test rank(pmucav) == 0
    @test size(pmucav) == 2
    @test is_root(pmucav)
    merge_histograms!(pmucav)
    @test all(ensemble(alg1).histogram.values .== [5.0, 5.0, 5.0, 5.0])
    @test all(ensemble(alg2).histogram.values .== [5.0, 5.0, 5.0, 5.0])
    ensemble(alg1).logweight.values .= [1.0, 2.0, 3.0, 4.0]
    distribute_logweight!(pmucav)
    @test all(ensemble(alg2).logweight.values .== [1.0, 2.0, 3.0, 4.0])

    if verbose
        println("ParallelMulticanonical tests completed")
    end

    return true
end

function test_parallel_tempering(; verbose=false)
    _ensure_mpi_init()

    backend = MPIBackend(MPI.COMM_WORLD)
    alg = Metropolis(MersenneTwister(10); β=0.8)
    pt = ParallelTempering(alg, backend; root=0)
    @test rank(pt) == 0
    @test size(pt) == 1
    @test is_root(pt)
    @test pt isa ParallelTemperingMessage
    @test pt.backend === backend
    @test pt.alg === alg
    @test index(pt) == 1
    @test isempty(pt.steps)
    @test isempty(pt.accepted)
    @test isempty(acceptance_rates(pt))
    @test acceptance_rate(pt) == 0.0

    update!(pt, -10.0)
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

    # local-vector mode
    v_algs = [Metropolis(MersenneTwister(11); β=1.0), Metropolis(MersenneTwister(12); β=0.5)]
    v_pt = ParallelTempering(v_algs)
    @test v_pt isa ParallelTemperingVector
    @test index(v_pt, 1) == 1
    update!(v_pt, [-10.0, -8.0])
    @test v_pt.stage == 1
    @test sum(v_pt.steps) >= 0

    v_pt2 = ParallelTempering([1.0, 0.5]; seed=123, rng=MersenneTwister)
    @test v_pt2 isa ParallelTemperingVector
    @test length(v_pt2.alg) == 2
    @test ensemble(v_pt2.alg[1]).beta == 1.0
    @test ensemble(v_pt2.alg[2]).beta == 0.5

    @test MonteCarloX.acceptance_rates([4, 0, 3], [1, 0, 3]) == [0.25, 0.0, 1.0]
    @test MonteCarloX.acceptance_rate([4, 0, 3], [1, 0, 3]) == 4 / 7

    @test MonteCarloX._resolve_pair(1, 0, 4) == (active=true, pair_id=1, partner_index=2)
    @test MonteCarloX._resolve_pair(2, 0, 4) == (active=true, pair_id=1, partner_index=1)
    @test MonteCarloX._resolve_pair(1, 1, 4) == (active=false, pair_id=0, partner_index=0)
    @test MonteCarloX._resolve_pair(3, 1, 4) == (active=true, pair_id=2, partner_index=2)

    alg_i = Metropolis(MersenneTwister(21); β=1.0)
    alg_j = Metropolis(MersenneTwister(22); β=0.5)
    x_i, x_j = 0.0, -5.0
    log_ratio = exchange_log_ratio(ensemble(alg_i), ensemble(alg_j), x_i, x_j)
    @test isapprox(log_ratio, 2.5; atol=1e-12)

    accepted = attempt_exchange_pair!(alg_i, alg_j, x_i, x_j, 0.0)
    @test accepted
    @test ensemble(alg_i).beta == 0.5
    @test ensemble(alg_j).beta == 1.0

    alg_i_reject = Metropolis(MersenneTwister(23); β=1.0)
    alg_j_reject = Metropolis(MersenneTwister(24); β=0.5)
    rejected = attempt_exchange_pair!(alg_i_reject, alg_j_reject, x_j, x_i, 1.0)
    @test !rejected
    @test ensemble(alg_i_reject).beta == 1.0
    @test ensemble(alg_j_reject).beta == 0.5

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
    @test interval == sweeps[index(v_pt3)]
    @test all(2 .<= sweeps .<= 50)
    @test sweeps[2] >= sweeps[1]

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
