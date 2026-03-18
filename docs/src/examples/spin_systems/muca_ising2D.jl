# %% #src
import Pkg                       #src
Pkg.activate(dirname(@__DIR__))  #src
Pkg.instantiate()                #src

# # Multicanonical Sampling of the 2D Ising Model
#
# Multicanonical (MUCA) sampling iteratively refines energy weights to produce
# a flat histogram, enabling efficient sampling across the phase transition.
# We validate against the exact density of states by
# [Beale (1996)](https://doi.org/10.1103/PhysRevLett.76.78).

# %%
using Random, StatsBase, Plots
using MonteCarloX, SpinSystems, MPI

# %%
# ## Parameters

L             = 8
n_iter        = 10
sweeps_therm  = 100
sweeps_record = 10_000

# %%
# ## Validation and visualization
#
# The exact log-density of states for the $L\times L$ Ising model (Beale 1996)
# serves as reference. `rmse_exact` quantifies convergence per iteration and
# `plot_muca` visualizes the histogram, estimated log-DOS against the exact
# reference, and the RMSE convergence.

exact_logdos          = logdos_exact_ising2D(L)
exact_logdos.weights .-= exact_logdos[0]
mask                  = .!isnan.(exact_logdos.weights)

function rmse_exact(lw)
    est = -deepcopy(lw.weights) .+ lw[0]
    return sqrt(mean((est[mask] .- exact_logdos.weights[mask]).^2))
end

function plot_muca(hists::Vector{BinnedObject}, lws::Vector{BinnedObject}; title="")
    n        = length(hists)
    cols     = palette(:viridis, max(n, 2))[1:n]
    energies = collect(hists[1].bins[1])
    i0       = findfirst(==(0), energies)

    p1 = plot(xlabel="E", ylabel="counts", title="$(title) histograms", legend=false)
    for i in 1:n
        plot!(p1, energies, hists[i].weights; lw=2, color=cols[i])
    end

    p2 = plot(xlabel="E", ylabel="-log w(E)", title="$(title) log-DOS", legend=false)
    for i in 1:n
        w = -deepcopy(lws[i].weights)
        i0 !== nothing && (w .-= w[i0])
        plot!(p2, energies, w; lw=2, color=cols[i])
    end
    ref = Float64.(exact_logdos.weights)
    i0 !== nothing && isfinite(ref[i0]) && (ref .-= ref[i0])
    plot!(p2, energies, ref; lw=2, color=:black, label="exact")

    rmse = rmse_exact.(lws)
    p3   = scatter(1:n, rmse; ms=4, color=:steelblue,
                   xlabel="iter", ylabel="RMSE",
                   title="$(title) convergence",
                   legend=false, yscale=:log10)

    return plot(p1, p2, p3; layout=(1,3), size=(980,260), margin=4Plots.mm), rmse
end

# %%
# ## Serial MUCA
#
# A single Markov chain accumulates the energy histogram. After each iteration
# weights are updated via the Wang-Landau rule until the histogram flattens.

sys = IsingLatticeOptim(L, L)
init!(sys, :random, rng=Xoshiro(1000))
alg = Multicanonical(Xoshiro(1000), get_centers(exact_logdos))

serial_hists, serial_lws = BinnedObject[], BinnedObject[]
for _ in 1:n_iter
    for _ in 1:(sweeps_therm  * length(sys.spins)); spin_flip!(sys, alg); end
    reset!(alg)
    for _ in 1:(sweeps_record * length(sys.spins)); spin_flip!(sys, alg); end
    update!(ensemble(alg); mode=:simple)
    push!(serial_hists, deepcopy(alg.ensemble.histogram))
    push!(serial_lws,   deepcopy(alg.ensemble.logweight))
end

plt, rmse = plot_muca(serial_hists, serial_lws; title="Serial")
println("Serial RMSE = ", round(last(rmse), digits=4))
plt

# %%
# ## Parallel MUCA with MPI
#
# Each rank runs an independent chain over `1/nranks` of the sweeps. After
# each iteration histograms are merged onto rank 0, weights updated and
# broadcast back. With a single rank (VS Code/REPL) the result is identical
# to the serial version above.

pmuca = ParallelMulticanonical(MPI.COMM_WORLD, root=0)
sys   = IsingLatticeOptim(L, L)
init!(sys, :random, rng=Xoshiro(1000 + pmuca.rank))
alg   = Multicanonical(Xoshiro(1000 + pmuca.rank), get_centers(exact_logdos))

mpi_hists, mpi_lws = BinnedObject[], BinnedObject[]
for _ in 1:n_iter
    for _ in 1:(sweeps_therm  * length(sys.spins));              spin_flip!(sys, alg); end
    reset!(alg)
    for _ in 1:(sweeps_record * length(sys.spins) / pmuca.size); spin_flip!(sys, alg); end
    merge_histograms!(pmuca, alg.ensemble.histogram)
    if is_root(pmuca)
        update!(ensemble(alg); mode=:simple)
        push!(mpi_hists, deepcopy(alg.ensemble.histogram))
        push!(mpi_lws,   deepcopy(alg.ensemble.logweight))
    end
    distribute_logweight!(pmuca, alg.ensemble.logweight)
end

if is_root(pmuca)
    plt, rmse = plot_muca(mpi_hists, mpi_lws; title="MPI")
    println("MPI RMSE = ", round(last(rmse), digits=4))
    plt
end

# %%
# ## Production runs
#
# For true parallelism, run the standalone MPI script in the same folder:
# ```bash
# mpiexec -n 4 julia --project=docs/src/examples \
#     docs/src/examples/spin_systems/muca_ising2D_mpi.jl
# ```