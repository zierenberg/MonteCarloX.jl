# Multi-rank replica-exchange (parallel tempering) over an MPIBackend.
#
# Run with 2 ranks:  mpiexec -n 2 julia --project test/mpi/test_replica_exchange_mpi.jl
# Launched automatically from test/runtests.jl when an MPI runtime is available.
#
# The single-process test suite can only build a 1-rank MPIBackend, so the actual cross-rank
# exchange path (_update_pair!, _exchange_packet_mpi, _partner_rank, MPI.Allgather) is never
# hit there. This file exercises it with ≥2 ranks and exits non-zero on any failed assertion.

using MPI
using MonteCarloX
using Random

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)
nranks >= 2 || error("this test needs at least 2 MPI ranks (got $nranks)")

pass = true
chk(cond, msg) = (cond || (println("rank $rank FAIL: $msg"); global pass = false); cond)

# One replica per rank on a β-ladder: rank 0 is cold (β=1.0), rank 1 hot (β=0.5).
betas = [1.0, 0.5]
alg = MetropolisAlgorithm(MersenneTwister(100 + rank); β = betas[rank + 1])
pt  = ParallelTempering(MPIBackend(comm), alg)

chk(pt isa MonteCarloX.ReplicaExchange{<:MonteCarloX.MPIBackend}, "MPI ReplicaExchange type")
chk(size(pt) == nranks, "size == nranks")
chk(index(pt) == rank + 1, "initial index == rank+1")

# ---- forced ACCEPT: (β_i−β_j)(x_i−x_j) > 0 makes log_ratio > 0, accepted regardless of u ----
# rank 0 (β=1.0) carries the higher energy, rank 1 (β=0.5) the lower → guaranteed swap.
x_local = rank == 0 ? 0.0 : -5.0
update!(pt, x_local)

# After the swap the two ranks trade ensembles (β) and the ladder permutation becomes [2, 1].
if rank == 0
    chk(ensemble(algorithm(pt)).beta == 0.5, "rank 0 β swapped to 0.5")
    chk(pt.steps[1] == 1,    "root counted one attempt")
    chk(pt.accepted[1] == 1, "root counted one acceptance")
else
    chk(ensemble(algorithm(pt)).beta == 1.0, "rank 1 β swapped to 1.0")
end
chk(sort(pt.indices) == collect(1:nranks), "ladder indices are a permutation")
chk(pt.indices == [2, 1],                  "ladder permuted to [2,1] after the swap")
chk(index(pt) == pt.indices[rank + 1],     "index() reads the local ladder slot")
chk(pt.stage == 1, "stage flipped to 1")

# ---- acceptance-rate reductions (root-only results, MPI.Reduce over ranks) ----
rates = acceptance_rates(pt)
if MonteCarloX.is_root(pt)
    chk(length(rates) == nranks - 1, "acceptance_rates length == n-1 on root")
    chk(all(0.0 .<= rates .<= 1.0),  "acceptance_rates in [0,1]")
    chk(acceptance_rate(pt) == 1.0,  "overall acceptance_rate == 1.0 (forced accept)")
else
    chk(isempty(rates), "acceptance_rates empty off-root")
end

# ---- run several more sweeps; the ladder must stay a valid permutation every step ----
for s in 1:6
    update!(pt, rank == 0 ? 0.0 : randn(alg.rng))
    chk(sort(pt.indices) == collect(1:nranks), "indices stay a permutation (sweep $s)")
end

# ---- reset! returns to the identity ladder / zero stats ----
reset!(pt)
chk(pt.stage == 0, "stage reset to 0")
chk(pt.indices == collect(1:nranks), "indices reset to identity")
chk(all(pt.steps .== 0) && all(pt.accepted .== 0), "counters reset")

# ---- global pass/fail: every rank must pass ----
all_pass = MPI.Allreduce(pass ? 1 : 0, MPI.PROD, comm) == 1
if MonteCarloX.is_root(pt)
    println(all_pass ? "MPI replica-exchange: ALL PASS ($nranks ranks)" :
                       "MPI replica-exchange: FAILURES")
end
MPI.Finalize()
exit(all_pass ? 0 : 1)
