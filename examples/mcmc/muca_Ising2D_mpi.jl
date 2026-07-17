"""
    muca_Ising2D_mpi.jl

Multicanonical (MUCA) sampling [Berg & Neuhaus 1992; Zierenberg et al. 2013] for 2D Ising model using MPI parallelism.
Run with: mpiexec -n 4 julia --project=docs docs/src/examples/spin_systems/muca_Ising2D_mpi.jl

Each MPI rank runs one independent replica and exchanges histograms via MPI.
"""

# %%                                                #src
import Pkg                                          #src
Pkg.activate(joinpath(@__DIR__, ".."))              #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

using Random, StatsBase, Plots
using MonteCarloX: update!
using MonteCarloX, MCXSpins
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
nothing #hide

# Setup
backend = init(:MPI)
alg = MulticanonicalAlgorithm(Xoshiro(1000 + rank(backend)), get_centers(exact_logdos))
pmuca = ParallelMulticanonical(backend, alg)

sys = IsingSystem([L, L])
init!(sys, :random, rng=alg.rng)

on_root(pmuca) do
    println("════════════════════════════════════════")
    println(" MPI Multicanonical Ising (L = $(L))    ")
    println(" Ranks = $(size(pmuca)), Iterations = $n_iter")
    println("════════════════════════════════════════")
    println("Starting simulation...")
end

# One sweep = one attempted flip per site (function barrier keeps the hot loop specialized)
sweep!(sys, alg, n_sweeps) =
    (for _ in 1:n_sweeps, _ in 1:length(sys.spins); spin_flip!(sys, alg); end)

# Main iteration loop
mpi_hists, mpi_lws = BinnedObject[], BinnedObject[]
for iter in 1:n_iter
    sweep!(sys, alg, sweeps_therm)
    reset!(alg)

    # Each rank does 1/nprocs of the total sweeps
    sweep!(sys, alg, sweeps_record ÷ size(pmuca))

    merge_histograms!(pmuca)

    on_root(pmuca) do
        update_logweight!(ensemble(alg); mode=:simple)
        rmse = rmse_exact(ensemble(alg).logweight)
        push!(mpi_hists, deepcopy(ensemble(alg).histogram))
        push!(mpi_lws, deepcopy(ensemble(alg).logweight))
    end

    distribute_logweight!(pmuca)
end

on_root(pmuca) do
    println("Simulation finished!")
    final_rmse = rmse_exact(mpi_lws[end])
    println("Final RMSE (vs exact Beale): $(round(final_rmse, digits=4))")
end
finalize!(backend)

# ## References
#
# - B. A. Berg, T. Neuhaus, *Multicanonical ensemble: a new approach to simulate first-order
#   phase transitions*, Phys. Rev. Lett. **68**, 9 (1992).
#   [doi:10.1103/PhysRevLett.68.9](https://doi.org/10.1103/PhysRevLett.68.9)
# - J. Zierenberg, M. Marenz, W. Janke, *Scaling properties of a parallel implementation of the
#   multicanonical algorithm*, Comput. Phys. Commun. **184**, 1155 (2013).
#   [doi:10.1016/j.cpc.2012.12.006](https://doi.org/10.1016/j.cpc.2012.12.006)
# - P. D. Beale, *Exact distribution of energies in the two-dimensional Ising model*,
#   Phys. Rev. Lett. **76**, 78 (1996).
#   [doi:10.1103/PhysRevLett.76.78](https://doi.org/10.1103/PhysRevLett.76.78)
