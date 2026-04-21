# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Checkpointing
#
# Demonstrate checkpoint/restore for a single-chain Monte Carlo simulation.
# We verify that a run interrupted at a checkpoint and then continued
# produces **identical** results to an uninterrupted reference run.
#
# This works because checkpointing serializes both the system state and the
# algorithm state (including the RNG), so the restored chain continues on
# the exact same trajectory.

using Random, Test
using MonteCarloX

# ## System definition
#
# Same double-well potential as in the parallel-chains example.

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

n_total  = CI_MODE ? 100 : 1000      # total number of samples
n_sweep  = 10                         # sweeps between samples
n_half   = n_total ÷ 2               # checkpoint after this many samples

seed = 42
β    = 2.0

# ## Reference run (uninterrupted)
#
# Run the full simulation in one go and record the time series.

ref_sys = System(0.0)
ref_alg = Metropolis(Xoshiro(seed); β=β)

ref_xs = Vector{Float64}(undef, n_total)
for j in 1:n_total
    for _ in 1:n_sweep; update!(ref_sys, ref_alg); end
    ref_xs[j] = ref_sys.x
end
println("Reference run: $(n_total) samples, final x = $(round(ref_sys.x; digits=6))")

# ## Checkpointed run
#
# Run the first half, checkpoint, destroy objects, restore, continue.

ckpt_dir  = mktempdir()
ckpt_file = joinpath(ckpt_dir, "ckpt.mcx")

# ### First half

sys = System(0.0)
alg = Metropolis(Xoshiro(seed); β=β)

ckpt = init_checkpoint(ckpt_file, (sys=sys, alg=alg, sample=0))

ckpt_xs = Vector{Float64}(undef, n_total)
for j in 1:n_half
    for _ in 1:n_sweep; update!(sys, alg); end
    ckpt_xs[j] = sys.x
end

checkpoint!(ckpt; sample=n_half)
println("Checkpoint written at sample $(n_half), x = $(round(sys.x; digits=6))")

# Verify first half already matches
@assert ckpt_xs[1:n_half] == ref_xs[1:n_half] "First half must match reference"

# ### Destroy and restore
#
# Simulate a restart: forget `sys` and `alg`, then restore from file.

ckpt  = restore_checkpoint(ckpt_file)
sys   = ckpt.sys
alg   = ckpt.alg
start = ckpt.sample + 1
println("Restored from checkpoint: sample=$(ckpt.sample), x = $(round(sys.x; digits=6))")

# ### Second half

for j in start:n_total
    for _ in 1:n_sweep; update!(sys, alg); end
    ckpt_xs[j] = sys.x
end

# Clean up checkpoint file after successful completion.
finalize!(ckpt)
println("Resumed run complete: $(n_total) samples, final x = $(round(sys.x; digits=6))")

# ## Verify identical trajectories

@assert ckpt_xs == ref_xs "Checkpointed run must exactly match reference run"
println("✓ Checkpointed trajectory is identical to uninterrupted reference.")

# ## Summary
#
# The checkpointing API:
# ```julia
# ckpt = init_checkpoint(file, (sys=sys, alg=alg, sweep=0))   # create session
# checkpoint!(ckpt; sweep=n)                                    # save state
# ckpt = restore_checkpoint(file)                               # restore session
# ckpt.sys, ckpt.alg, ckpt.sweep                               # access fields
# finalize!(ckpt)                                               # clean up
# ```
