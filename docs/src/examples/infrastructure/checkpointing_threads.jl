# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Checkpointing with Parallel Chains (Threads)
#
# Demonstrate checkpoint/restore for a multi-chain simulation using
# threads.  We verify that a run interrupted at a checkpoint and then
# continued produces **identical** results to an uninterrupted reference.
#
# Launch with:
# ```bash
# julia --threads=4 --project docs/src/examples/infrastructure/checkpointing_threads.jl
# ```
#
# For the MPI version, see `checkpointing_mpi.jl`.

using Random, Test
using MonteCarloX

# ## System definition

mutable struct System
    x::Float64
end

E(sys::System) = 1/4*(sys.x^2 - 2)^2

function update!(sys::System, alg::AbstractImportanceSampling; delta=0.1)
    x_new = sys.x + delta * randn(alg.rng)
    accept!(alg, E(System(x_new)), E(sys)) && (sys.x = x_new)
end

# ## Parameters

const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

n_total = CI_MODE ? 100 : 1000
n_sweep = 10
n_half  = n_total ÷ 2

seed = 42
β    = 2.0

backend = ThreadsBackend()

# ## Checkpoint directory
#
# On HPC, set `MCX_RUN_DIR` to a job-specific path (e.g. containing
# `SLURM_JOB_ID`).  Falls back to a temporary directory.

run_dir   = get(ENV, "MCX_RUN_DIR", mktempdir())
ckpt_file = joinpath(run_dir, "ckpt.mcx")

# ## Reference run (uninterrupted)

ref_algs = [Metropolis(Xoshiro(seed + i); β=β) for i in 1:size(backend)]
ref_pc   = ParallelChains(backend, ref_algs)
ref_sys  = [System(0.0) for _ in 1:size(backend)]
ref_xs   = zeros(Float64, size(backend), n_total)

with_parallel(ref_pc) do i, alg
    sys = ref_sys[i]
    for j in 1:n_total
        for _ in 1:n_sweep; update!(sys, alg); end
        ref_xs[i, j] = sys.x
    end
end

println("Reference run: $(size(backend)) chains × $(n_total) samples")

# ## Checkpointed run
#
# With threads, all chains live in shared memory.  We checkpoint the
# full `ParallelChains` object and the vector of systems in one call.

# ### First half

algs = [Metropolis(Xoshiro(seed + i); β=β) for i in 1:size(backend)]
pc   = ParallelChains(backend, algs)
sys  = [System(0.0) for _ in 1:size(backend)]

ckpt    = init_checkpoint(ckpt_file, (sys=sys, pc=pc, sample=0))
ckpt_xs = zeros(Float64, size(backend), n_total)

with_parallel(pc) do i, alg
    s = sys[i]
    for j in 1:n_half
        for _ in 1:n_sweep; update!(s, alg); end
        ckpt_xs[i, j] = s.x
    end
end

checkpoint!(ckpt; sample=n_half)
println("Checkpoint written at sample $(n_half)")

@assert ckpt_xs[:, 1:n_half] == ref_xs[:, 1:n_half] "First half must match reference"

# ### Destroy and restore

ckpt  = restore_checkpoint(ckpt_file)
pc    = ckpt.pc
sys   = ckpt.sys
start = ckpt.sample + 1

println("Restored from checkpoint: sample=$(ckpt.sample)")

# ### Second half

with_parallel(pc) do i, alg
    s = sys[i]
    for j in start:n_total
        for _ in 1:n_sweep; update!(s, alg); end
        ckpt_xs[i, j] = s.x
    end
end

finalize!(ckpt)
println("Resumed run complete: $(size(backend)) chains × $(n_total) samples")

# ## Verify identical trajectories

@assert ckpt_xs == ref_xs "Checkpointed run must exactly match reference run"
println("✓ Checkpointed trajectory is identical to uninterrupted reference.")
