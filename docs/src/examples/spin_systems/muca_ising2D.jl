# # Multicanonical Sampling of the 2D Ising Model
#
# This example demonstrates multicanonical (MUCA) sampling of the 2D Ising
# model using **MonteCarloX.jl**. We compare a serial implementation against
# a parallel MPI version, both recovering the exact density of states due to
# [Beale (1996)](https://doi.org/10.1103/PhysRevLett.76.78).

# !!! note "Also available as a Jupyter notebook"
#     This example is also available as a
#     [Jupyter notebook](https://github.com/zierenberg/MonteCarloX.jl/blob/gh-pages/dev/generated/muca_ising2D.ipynb).

# %%
# ## Setup

#nb import Pkg
#nb Pkg.activate(dirname(@__DIR__))
#nb Pkg.instantiate()

using Random
using StatsBase
using Plots
using MonteCarloX
using SpinSystems
using MPI

if !MPI.Initialized()
    required = MPI.THREAD_FUNNELED
    provided = MPI.Init_thread(required)
    if provided < required
        error("MPI thread support insufficient: required THREAD_FUNNELED, " *
              "got level $(provided).")
    end
end

# %%
# ## System parameters
#
# We simulate an $L \times L$ Ising lattice. The multicanonical algorithm
# iterates `n_iter` times, each consisting of `sweeps_therm` thermalization
# sweeps followed by `sweeps_record` measurement sweeps.

L = 8
n_iter        = 10
sweeps_therm  = 100
sweeps_record = 10_000
domain        = (-2*L^2):(4):(0)

# Load the exact Beale logDOS for RMSE reference, normalized at E=0
exact_logdos = logdos_exact_ising2D(L)
exact_logdos.weights .-= exact_logdos[0]
mask_rmse    = .!isnan.(exact_logdos.weights)
exact_masked = exact_logdos.weights[mask_rmse]

function rmse_exact(lw)
    MonteCarloX._assert_same_domain(lw, exact_logdos)
    est  = -deepcopy(lw.weights)
    est .+= lw[0]
    return sqrt(mean((est[mask_rmse] .- exact_masked).^2))
end

# %%
# ## Serial MUCA
#
# In the serial version, a single Markov chain accumulates the energy
# histogram. After each iteration the weights are updated using the
# simple Wang-Landau rule via `update!(ensemble(alg); mode=:simple)`.

sys = IsingLatticeOptim(L, L)
init!(sys, :random, rng=Xoshiro(1000))
alg = Multicanonical(Xoshiro(1000), domain[1]:4:domain[end])

serial_hists = BinnedObject[]
serial_lws   = BinnedObject[]

for _ in 1:n_iter
    for _ in 1:(sweeps_therm * length(sys.spins))
        spin_flip!(sys, alg)
    end
    reset!(alg)
    for _ in 1:(sweeps_record * length(sys.spins))
        spin_flip!(sys, alg)
    end
    update!(ensemble(alg); mode=:simple)
    push!(serial_hists, deepcopy(alg.ensemble.histogram))
    push!(serial_lws,   deepcopy(alg.ensemble.logweight))
end

# After the iteration we compare the recovered log-density of states against
# the exact Beale result and report the RMSE over iterations.

plt, rmse = plot_from_hist_lw(serial_hists, serial_lws; title_prefix="Serial ")
println("Final serial RMSE (exact) = ", round(last(rmse), digits=4))
plt

# %%
# ## Parallel MUCA with MPI
#
# The parallel version distributes the measurement sweeps across MPI ranks.
# After each iteration, all ranks merge their histograms onto rank 0 via
# `merge_histograms!`, which updates the weights and broadcasts them back
# with `distribute_logweight!`. When running inside this notebook only
# 1 MPI rank is active, so the result is identical to the serial case.

pmuca = ParallelMulticanonical(MPI.COMM_WORLD, root=0)
sys   = IsingLatticeOptim(L, L)
init!(sys, :random, rng=Xoshiro(1000 + pmuca.rank))
alg   = Multicanonical(Xoshiro(1000 + pmuca.rank), domain[1]:4:domain[end])

mpi_hists = BinnedObject[]
mpi_lws   = BinnedObject[]

for _ in 1:n_iter
    for _ in 1:(sweeps_therm * length(sys.spins))
        spin_flip!(sys, alg)
    end
    reset!(alg)
    for _ in 1:(sweeps_record * length(sys.spins) / pmuca.size)
        spin_flip!(sys, alg)
    end
    merge_histograms!(pmuca, alg.ensemble.histogram)
    if is_root(pmuca)
        update!(ensemble(alg); mode=:simple)
        push!(mpi_hists, deepcopy(alg.ensemble.histogram))
        push!(mpi_lws,   deepcopy(alg.ensemble.logweight))
    end
    distribute_logweight!(pmuca, alg.ensemble.logweight)
end

if is_root(pmuca)
    plt, rmse = plot_from_hist_lw(mpi_hists, mpi_lws; title_prefix="MPI ")
    println("Final MPI RMSE (exact) = ", round(last(rmse), digits=4))
    plt
end

# %%
# ## Production runs with multiple MPI ranks
#
# The notebook above runs with a single MPI rank. For production runs with
# true parallelism, use the standalone script in the same folder:
#
# ```bash
# mpiexec -n 4 julia --project=docs/src/examples docs/src/examples/spin_systems/muca_ising2D_mpi.jl
# ```