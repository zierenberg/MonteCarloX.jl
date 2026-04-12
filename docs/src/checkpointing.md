# Checkpointing

Long-running simulations — especially on HPC clusters with job time limits — need
to save their state and resume after an interruption.

MonteCarloX provides a simple checkpoint API based on Julia's `Serialization`.
A `CheckpointSession` tracks a file path and a set of linked objects that are
serialized together on every `checkpoint!` call.

## Lifecycle

```julia
using MonteCarloX

# 1. Create session — writes an initial checkpoint
ckpt = init_checkpoint("run/ckpt.mcx", (sys=sys, alg=alg); sweep=0)

# 2. Inside the loop — write rolling checkpoints
checkpoint!(ckpt; sweep=sweep)

# 3. On restart — restore state
state = restore("run/ckpt.mcx")
sys = state.sys
alg = state.alg
start_sweep = state.sweep + 1
relink!(ckpt, (sys=sys, alg=alg))

# 4. After successful completion — clean up
finalize!(ckpt)
```

The checkpoint file is written atomically (tmp + rename) so a crash during
write never corrupts the previous checkpoint.

## HPC example: checkpoint and resume

```julia
using Random, MonteCarloX

checkpoint_file = "job/ckpt.mcx"

if isfile(checkpoint_file)
    state = restore(checkpoint_file)
    sys = state.sys
    alg = state.alg
    start_sweep = state.sweep + 1
    ckpt = init_checkpoint(checkpoint_file, (sys=sys, alg=alg); sweep=state.sweep)
else
    rng = MersenneTwister(42)
    alg = Metropolis(rng; β=0.44)
    sys = MySystem()
    start_sweep = 1
    ckpt = init_checkpoint(checkpoint_file, (sys=sys, alg=alg); sweep=0)
end

x = 0.0
for sweep in start_sweep:n_sweeps
    x_new = x + randn(alg.rng)
    x = accept!(alg, x_new, x) ? x_new : x

    if sweep % 100_000 == 0
        checkpoint!(ckpt; sweep=sweep)
    end
end

finalize!(ckpt)
```

## Continuous-time loops

`advance!` accepts optional `ckpt` and `checkpoint_interval` keyword arguments:

```julia
ckpt = init_checkpoint("kmc/ckpt.mcx", (alg=alg, sys=sys); step=0)
advance!(alg, sys, 1e8; ckpt=ckpt, checkpoint_interval=1000)
```

## API reference

```@docs
CheckpointSession
init_checkpoint
checkpoint!
restore
relink!
finalize!
```
