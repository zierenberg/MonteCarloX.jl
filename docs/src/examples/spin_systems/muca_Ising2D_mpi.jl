"""
    muca_Ising2D_mpi.jl

Multicanonical (MUCA) sampling for 2D Ising model using MPI parallelism.
Run with: mpiexec -n 4 julia --project=docs docs/src/examples/spin_systems/muca_Ising2D_mpi.jl

Each MPI rank runs one independent replica and exchanges histograms via MPI.
"""

# %%                                                #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

using Random, StatsBase, Plots
using MonteCarloX: update!
using MonteCarloX, SpinSystems
using MPI

# Parameters
L             = 8
n_iter        = 10
sweeps_therm  = 1_000
sweeps_record = 100_000

# Validation and visualization
exact_logdos          = logdos_exact_ising2D(L)
exact_logdos.values .-= exact_logdos[0]
mask                  = .!isnan.(exact_logdos.values)

function rmse_exact(lw)
    est = -deepcopy(lw.values) .+ lw[0]
    return sqrt(mean((est[mask] .- exact_logdos.values[mask]).^2))
end

# Setup
backend = init(:MPI)
alg = Multicanonical(Xoshiro(1000 + rank(backend)), get_centers(exact_logdos))
pmuca = ParallelMulticanonical(backend, alg, root=0)

sys = Ising([L, L])
init!(sys, :random, rng=alg.rng)

if is_root(pmuca)
    println("════════════════════════════════════════")
    println(" MPI Multicanonical Ising (L = $(L))    ")
    println(" Ranks = $(size(pmuca)), Iterations = $n_iter")
    println("════════════════════════════════════════")
    println("Starting simulation...")
end

# Main iteration loop
mpi_hists, mpi_lws = BinnedObject[], BinnedObject[]
for iter in 1:n_iter
    # Thermalization
    for _ in 1:(sweeps_therm * length(sys.spins))
        spin_flip!(sys, alg)
    end
    reset!(alg)
    
    # Each rank does 1/nprocs of the total sweeps
    for _ in 1:(sweeps_record * length(sys.spins) / size(pmuca))
        spin_flip!(sys, alg)
    end

    merge_histograms!(pmuca)
    
    if is_root(pmuca)
        MonteCarloX.update!(ensemble(alg); mode=:simple)
        rmse = rmse_exact(ensemble(alg).logweight)
        push!(mpi_hists, deepcopy(ensemble(alg).histogram))
        push!(mpi_lws, deepcopy(ensemble(alg).logweight))
    end
    
    distribute_logweight!(pmuca)
end

if is_root(pmuca)
    println("Simulation finished!")
    final_rmse = rmse_exact(mpi_lws[end])
    println("Final RMSE (vs exact Beale): $(round(final_rmse, digits=4))")
end
finalize!(backend)