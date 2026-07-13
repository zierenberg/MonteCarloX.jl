# # Parallel Tempering (MPI) for 2D Ising
#
# Run with:
# mpiexec -n 4 julia --project=docs docs/src/examples/spin_systems/pt_Ising2D_mpi.jl
#
# One MPI rank hosts one replica. The replica exchange protocol coordinates
# neighbor swaps via MPI, with energy swap observables passed to the PT
# coordinator. Only scalar data is exchanged; no full system state is transferred.

# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

using Random, Statistics, StatsBase, Plots
using MonteCarloX: update!
using MonteCarloX, MCXSpins, MPI

MPI.Initialized() || MPI.Init()

# Parameters

L                       = 8
nreplicas               = 4
nmeasurements           = 20_000
ntherm_init             = 2_000
sweeps_between_exchange = 200

Tmin = 1.5
Tmax = 3.0
seed = 42

# Sweep helper

function sweep_replica!(sys, alg, L)
    for _ in 1:(L * L)
        spin_flip!(sys, alg)
    end
    return nothing
end

# Setup

backend = init(:MPI)
nranks = size(backend)
nranks >= 2 || throw(ArgumentError("MPI PT requires at least 2 ranks"))

betas = set_betas(nranks, inv(Tmax), inv(Tmin), :uniform)
pt = ParallelTempering(betas; seed=seed, rng=Xoshiro, backend=backend)
alg = algorithm(pt)

sys = IsingSystem([L, L])
init!(sys, :random; rng=alg.rng)

on_root(pt) do
    println("════════════════════════════════════════")
    println(" PT Ising2D MPI (L = $(L))              ")
    println(" Ranks = $(nranks), Measurements = $(nmeasurements)")
    println("════════════════════════════════════════")
    println("Thermalization...")
end

for _ in 1:ntherm_init
    sweep_replica!(sys, alg, L)
end

on_root(pt) do
    println("Starting measurements...")
end

# Main measurement loop

local_samples = Tuple{Int,Float64}[]
for meas in 1:nmeasurements
    sweep_replica!(sys, alg, L)

    e = energy(sys)
    push!(local_samples, (index(pt), e))

    if meas % sweeps_between_exchange == 0
        MonteCarloX.update!(pt, e)
    end
end

on_root(pt) do
    println("Gathering results...")
end

all_samples = MPI.gather(local_samples, backend.comm; root=backend.root)
rates = acceptance_rates(pt)

on_root(pt) do
    println("Simulation finished!")
    println("\nExchange acceptance rates:")
    for (i, acc_rate) in enumerate(rates)
        println("  (β=$(round(betas[i], digits=3)), β=$(round(betas[i+1], digits=3))): $(round(acc_rate, digits=3))")
    end

    println("\nMean energy comparison:")
    energy_samples = [Float64[] for _ in 1:nranks]
    for rank_samples in all_samples
        for (i, e) in rank_samples
            push!(energy_samples[i], e)
        end
    end
    logdos = logdos_exact_ising2D(L)
    E_exact = get_centers(logdos)
    for (i, β) in enumerate(betas)
        samples = energy_samples[i]
        mean_exact = mean(E_exact, weights(reweight(logdos, -β .* E_exact)))
        mean_measured = isempty(samples) ? NaN : mean(samples)
        delta_mean = abs(mean_measured - mean_exact)
        println("  β=$(round(β, digits=3)): <E>_meas=$(round(mean_measured, digits=1)) vs <E>_exact=$(round(mean_exact, digits=1)) Δ=$(round(delta_mean, digits=2))")
    end
end

finalize!(backend)
