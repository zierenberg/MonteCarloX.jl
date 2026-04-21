# %%                                                #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src

# # Checkpoint And Restart For The 2D Ising Model
#
# This example shows a minimal, practical checkpoint workflow:
# 1. run Metropolis sweeps and write one rolling checkpoint file,
# 2. simulate an interruption,
# 3. restore from the checkpoint and continue,
# 4. remove the checkpoint file after successful completion.

using Random
using MonteCarloX, SpinSystems

const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

# -----------------------------------------------------------------------------
# Simulation setup
# -----------------------------------------------------------------------------

L = 16
beta = 0.44
total_sweeps = CI_MODE ? 120 : 5_000
checkpoint_interval = CI_MODE ? 20 : 500
report_interval = CI_MODE ? 20 : 500
interrupt_after = CI_MODE ? 60 : 2_500

# Persistent checkpoint location.
# On HPC, set MCX_CHECKPOINT_BASE to a shared filesystem path
# (for example under SCRATCH) so restart works on any node.
checkpoint_base = get(
    ENV,
    "MCX_CHECKPOINT_BASE",
    joinpath(pwd(), "checkpoints", "ising2d", "ckpt"),
)
checkpoint_file = checkpoint_base * ".mcx"

sys = Ising([L, L])
alg = Metropolis(Xoshiro(2026); β=beta)
ckpt = init_checkpoint(checkpoint_file, (sys=sys, alg=alg, sweep=0))

function run_sweeps!(sys, alg, sweep_start::Int, sweep_stop::Int;
                     ckpt::CheckpointSession,
                     checkpoint_interval::Int,
                     report_interval::Int)
    for sweep in sweep_start:sweep_stop
        for _ in 1:length(sys.spins)
            spin_flip!(sys, alg)
        end

        if sweep % checkpoint_interval == 0
            checkpoint!(ckpt; sweep=sweep)
        end

        if sweep % report_interval == 0 || sweep == sweep_stop
            println("sweep=$(sweep), E=$(energy(sys)), M=$(magnetization(sys)), accepted=$(alg.accepted)")
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Loop 1: run and create checkpoints (simulated preemption)
# -----------------------------------------------------------------------------

init!(sys, :random, rng=Xoshiro(2026))
println("Phase 1: running until simulated interruption at sweep $(interrupt_after)")
run_sweeps!(
    sys,
    alg,
    1,
    interrupt_after;
    ckpt=ckpt,
    checkpoint_interval=checkpoint_interval,
    report_interval=report_interval,
)

if !isfile(checkpoint_file)
    throw(ArgumentError(
        "No checkpoint file found at $(checkpoint_file). " *
        "Choose checkpoint_interval <= interrupt_after."))
end

# -----------------------------------------------------------------------------
# Loop 2: restore and continue
# -----------------------------------------------------------------------------

println("Phase 2: restoring and continuing to sweep $(total_sweeps)")
ckpt = restore_checkpoint(checkpoint_file)
sys = ckpt.sys
alg = ckpt.alg
start_sweep = ckpt.sweep + 1

if start_sweep <= total_sweeps
    run_sweeps!(
        sys,
        alg,
        start_sweep,
        total_sweeps;
        ckpt=ckpt,
        checkpoint_interval=checkpoint_interval,
        report_interval=report_interval,
    )
end

# Simulation finished successfully, checkpoint no longer needed.
finalize!(ckpt)

println("Finished at sweep=$(alg.steps), final energy=$(energy(sys))")
println("Checkpoint file removed: $(checkpoint_file)")
