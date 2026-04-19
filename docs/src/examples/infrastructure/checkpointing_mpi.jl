# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Checkpointing with Parallel Chains (MPI)
#
# Demonstrate checkpoint/restore for a multi-chain simulation using MPI.
# Each rank writes its own checkpoint file.  We verify that a run
# interrupted at a checkpoint and then continued produces **identical**
# results to an uninterrupted reference.
#
# Launch with:
# ```bash
# mpiexec -n 4 julia --project docs/src/examples/infrastructure/checkpointing_mpi.jl
# ```
#
# For the threads version, see `checkpointing_threads.jl`.
#
# Key difference from threads: each rank checkpoints independently to its
# own file.  The `MPI.Comm` handle is not serializable, so we checkpoint
# the rank-local `sys` and `alg` and reconstruct `ParallelChains` from a
# fresh backend on restore.

using Random, Test
using MonteCarloX, MPI

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

backend = init(:MPI)

# ## Checkpoint directory
#
# On HPC, set `MCX_RUN_DIR` to a job-specific path (e.g. containing
# `SLURM_JOB_ID`).  Each rank writes its own checkpoint file.

run_dir   = get(ENV, "MCX_RUN_DIR", tempdir())
ckpt_file = joinpath(run_dir, "ckpt_rank$(rank(backend)).mcx")

# ## Reference run (uninterrupted)

ref_alg = Metropolis(Xoshiro(seed + rank(backend) + 1); β=β)
ref_pc  = ParallelChains(backend, ref_alg)
ref_sys = System(0.0)
ref_xs  = Vector{Float64}(undef, n_total)

with_parallel(ref_pc) do alg
    sys = ref_sys
    for j in 1:n_total
        for _ in 1:n_sweep; update!(sys, alg); end
        ref_xs[j] = sys.x
    end
end

on_root(ref_pc) do
    println("Reference run: $(size(backend)) ranks × $(n_total) samples")
end

# ## Checkpointed run
#
# With MPI, each rank owns one `sys` and one `alg`.  Each rank writes its
# own checkpoint file.  The `ParallelChains` wrapper is reconstructed from
# a fresh backend on restore (MPI communicators are not serializable).

# ### First half

alg = Metropolis(Xoshiro(seed + rank(backend) + 1); β=β)
pc  = ParallelChains(backend, alg)
sys = System(0.0)

ckpt    = init_checkpoint(ckpt_file, (sys=sys, alg=alg, sample=0))
ckpt_xs = Vector{Float64}(undef, n_total)

with_parallel(pc) do alg
    for j in 1:n_half
        for _ in 1:n_sweep; update!(sys, alg); end
        ckpt_xs[j] = sys.x
    end
end

checkpoint!(ckpt; sample=n_half)

on_root(pc) do
    println("Checkpoint written at sample $(n_half)")
end

@assert ckpt_xs[1:n_half] == ref_xs[1:n_half] "First half must match reference on rank $(rank(backend))"

# ### Destroy and restore
#
# Each rank restores its own checkpoint file independently.
# The `ParallelChains` wrapper is reconstructed from the existing backend.

ckpt  = restore_checkpoint(ckpt_file)
sys   = ckpt.sys
alg   = ckpt.alg
start = ckpt.sample + 1
pc    = ParallelChains(backend, alg)

on_root(pc) do
    println("Restored from checkpoint: sample=$(ckpt.sample)")
end

# ### Second half

with_parallel(pc) do alg
    for j in start:n_total
        for _ in 1:n_sweep; update!(sys, alg); end
        ckpt_xs[j] = sys.x
    end
end

finalize!(ckpt)

on_root(pc) do
    println("Resumed run complete: $(size(backend)) ranks × $(n_total) samples")
end

# ## Verify identical trajectories
#
# Each rank verifies its own trajectory independently.

@assert ckpt_xs == ref_xs "Checkpointed run must exactly match reference on rank $(rank(backend))"

on_root(pc) do
    println("✓ Checkpointed trajectory is identical to uninterrupted reference.")
end

finalize!(backend)
