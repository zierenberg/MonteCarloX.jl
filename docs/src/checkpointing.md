# Checkpointing

Long-running simulations — especially on HPC clusters with job time limits — need
to save their state and resume after an interruption.

MonteCarloX provides a simple checkpoint API based on Julia's `Serialization`.
A `CheckpointSession` tracks a file path and a set of linked objects that are
serialized together on every `checkpoint!` call.

Checkpointing distinguishes between linked objects passed as named tuples upon init 
(tracked throughout) and meta data provided at each checkpoint added as kwargs. 

## Lifecycle

```julia
using MonteCarloX

# 1. Create session — writes an initial checkpoint 
ckpt = init_checkpoint("ckpt.mcx", (sys=sys, alg=alg), sweep=0)

# 2. Inside the loop — write rolling checkpoints
checkpoint!(ckpt; sweep=sweep)

# 3. On restart — restore state
state = restore_checkpoint("run/ckpt.mcx")
start_sweep = state.sweep

# 4. After successful completion — clean up
finalize!(ckpt)
```

The checkpoint file is written atomically (tmp + rename) so a crash during
write never corrupts the previous checkpoint.

## Single-chain example

```julia
using Random, Test
using MonteCarloX

# Define system
mutable struct System
    x::Float64
end
E(sys::System) = 1/4*(sys.x^2 - 2)^2

function update!(sys::System, alg::AbstractImportanceSampling; delta=0.1)
    x_new = sys.x + delta * randn(alg.rng)
    accept!(alg, E(System(x_new)), E(sys)) && (sys.x = x_new)
end

# Parameters
n_total = 1000
n_sweep = 10
n_half  = n_total ÷ 2
seed    = 42
β       = 2.0

# Run first half
sys = System(0.0)
alg = Metropolis(Xoshiro(seed); β=β)
ckpt = init_checkpoint("ckpt.mcx", (sys=sys, alg=alg, sample=0))

for j in 1:n_half
    for _ in 1:n_sweep; update!(sys, alg); end
end

checkpoint!(ckpt; sample=n_half)

# Restore and continue
ckpt  = restore_checkpoint("ckpt.mcx")
sys   = ckpt.sys
alg   = ckpt.alg
start = ckpt.sample + 1

for j in start:n_total
    for _ in 1:n_sweep; update!(sys, alg); end
end

finalize!(ckpt)
```

## Parallel chains with threads

With thread-based parallelization, all chains share memory, so you can
checkpoint the entire `ParallelChains` state plus system vector in one call:

```julia
using Random, MonteCarloX

backend = ThreadsBackend()
algs    = [Metropolis(Xoshiro(42 + i); β=2.0) for i in 1:size(backend)]
pc      = ParallelChains(backend, algs)
sys     = [System(0.0) for _ in 1:size(backend)]

# Create checkpoint session
ckpt = init_checkpoint("ckpt.mcx", (sys=sys, pc=pc, sample=0))

# Inside simulation loop
checkpoint!(ckpt; sample=0)

# On restart
ckpt  = restore_checkpoint("ckpt.mcx")
pc    = ckpt.pc
sys   = ckpt.sys
start = ckpt.sample + 1

finalize!(ckpt)
```

## Parallel chains with MPI

Each MPI rank writes its own checkpoint file. The `ParallelChains` object
must be reconstructed from a fresh backend on restore (MPI communicators
are not serializable):

```julia
using Random, MonteCarloX, MPI

backend = init(:MPI)
alg     = Metropolis(Xoshiro(42 + rank(backend)); β=2.0)
pc      = ParallelChains(backend, alg)
sys     = System(0.0)

# Each rank writes its own file
ckpt_file = joinpath("./", "ckpt_rank$(rank(backend)).mcx")
ckpt = init_checkpoint(ckpt_file, (sys=sys, alg=alg, sample=0))

# Restore on restart
ckpt  = restore_checkpoint(ckpt_file)
sys   = ckpt.sys
alg   = ckpt.alg
pc    = ParallelChains(backend, alg)  # reconstruct from existing backend
start = ckpt.sample + 1

finalize!(ckpt)
```

## HPC checkpoint directory

Set `MCX_RUN_DIR` to a job-specific path (e.g. containing `SLURM_JOB_ID`):

```julia
run_dir   = get(ENV, "MCX_RUN_DIR", mktempdir())
ckpt_file = joinpath(run_dir, "ckpt.mcx")
```

## API reference

```@docs
CheckpointSession
init_checkpoint
checkpoint!
restore_checkpoint
finalize!
```
