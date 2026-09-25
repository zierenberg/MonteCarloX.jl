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
chk(ensemble_index(pt) == rank + 1, "initial index == rank+1")

# ---- forced ACCEPT: (β_i−β_j)(x_i−x_j) > 0 makes log_ratio > 0, accepted regardless of u ----
# rank 0 (β=1.0) carries the higher energy, rank 1 (β=0.5) the lower → guaranteed swap.
x_local = rank == 0 ? 0.0 : -5.0
attempt_exchange!(pt, x_local)

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
chk(ensemble_index(pt) == pt.indices[rank + 1],     "ensemble_index() reads the local ladder slot")
chk(pt.stage == 1, "stage flipped to 1")

# ---- acceptance-rate reductions (root-only results, MPI.Reduce over ranks) ----
# Both queries are COLLECTIVE (MPI.Reduce): every rank must call them, even though only the root
# gets a meaningful answer. Calling one inside an `is_root` branch deadlocks the other rank.
rates = acceptance_rates(pt)
overall = acceptance_rate(pt)
if MonteCarloX.is_root(pt)
    chk(length(rates) == nranks - 1, "acceptance_rates length == n-1 on root")
    chk(all(0.0 .<= rates .<= 1.0),  "acceptance_rates in [0,1]")
    chk(overall == 1.0,              "overall acceptance_rate == 1.0 (forced accept)")
else
    chk(isempty(rates), "acceptance_rates empty off-root")
end

# ---- run several more sweeps; the ladder must stay a valid permutation every step ----
for s in 1:6
    attempt_exchange!(pt, rank == 0 ? 0.0 : randn(alg.rng))
    chk(sort(pt.indices) == collect(1:nranks), "indices stay a permutation (sweep $s)")
end

# ---- reset! clears statistics only; the ladder permutation is live state ----
ladder_before, stage_before = copy(pt.indices), pt.stage
reset!(pt)
chk(all(pt.steps .== 0) && all(pt.accepted .== 0), "counters reset")
chk(pt.indices == ladder_before && pt.stage == stage_before, "reset! leaves the ladder alone")

# ---- tabulated ensembles: the three-phase protocol must deliver the whole table ----
# Boltzmann ensembles are 8 bytes, so the exchanges above never exercise the part that matters for
# multicanonical / Wang-Landau: the ensemble carries logweight and histogram ARRAYS, sent only in
# phase 3, on acceptance. Distinguishable tables per rank show whether the right one arrived.
muca = MulticanonicalEnsemble(0.0:1.0:9.0; init = (rank == 0 ? 10.0 : 20.0))
muca_rx = ReplicaExchange(MPIBackend(comm),
                          MetropolisHastingsAlgorithm(Xoshiro(7 + rank), muca, MetropolisBalance()))
table() = get_values(ensemble(algorithm(muca_rx)).logweight)
nbins = length(table())

for s in 1:4   # flat logweights ⇒ log_ratio == 0 ⇒ accepted, so the table moves every round
    attempt_exchange!(muca_rx, float(rank))
    chk(length(table()) == nbins, "muca table length preserved (round $s)")
    chk(all(table() .== (ensemble_index(muca_rx) == 2 ? 20.0 : 10.0)),
        "muca table matches ensemble_index (round $s)")
end

# ---- global pass/fail: every rank must pass ----
all_pass = MPI.Allreduce(pass ? 1 : 0, MPI.PROD, comm) == 1
if MonteCarloX.is_root(pt)
    println(all_pass ? "MPI replica-exchange: ALL PASS ($nranks ranks)" :
                       "MPI replica-exchange: FAILURES")
end
MPI.Finalize()
exit(all_pass ? 0 : 1)
