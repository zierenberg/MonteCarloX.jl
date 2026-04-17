# %%                                                #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Multicanonical sampling of the 2D Ising model
#
# Multicanonical (muca) sampling iteratively refines energy weights to produce
# a flat histogram, enabling efficient sampling across the phase transition.
# We validate against the exact density of states by
# [Beale (1996)](https://doi.org/10.1103/PhysRevLett.76.78).

using Random, StatsBase, Plots
using MonteCarloX, SpinSystems

# ## Parameters
const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

L             = 8;
n_iter        = CI_MODE ? 3 : 10;
sweeps_therm  = CI_MODE ? 100 : 1_000;
sweeps_record = CI_MODE ? 1_000 : 100_000;

# ## Validation and visualization
#
# The exact log-density of states for the ``L\times L`` Ising model (Beale 1996)
# serves as reference. `rmse_exact` quantifies convergence per iteration and
# `plot_muca` visualizes the histogram, estimated log-DOS against the exact
# reference, and the RMSE convergence.

exact_logdos          = logdos_exact_ising2D(L)
exact_logdos.values .-= exact_logdos[0]
mask                  = .!isnan.(exact_logdos.values)

function rmse_exact(lw)
    est = -deepcopy(lw.values) .+ lw[0]
    return sqrt(mean((est[mask] .- exact_logdos.values[mask]).^2))
end

function plot_muca(hists::Vector{BinnedObject}, lws::Vector{BinnedObject}; title="")
    n        = length(hists)
    cols     = palette(:viridis, max(n, 2))[1:n]
    energies = get_centers(hists[1])
    i0       = findfirst(==(0), energies)

    p1 = plot(xlabel="E", ylabel="counts", title="$(title) histograms", legend=false)
    for i in 1:n
        plot!(p1, energies, hists[i].values; lw=2, color=cols[i])
    end

    p2 = plot(xlabel="E", ylabel="-log w(E)", title="$(title) log-DOS", legend=false)
    for i in 1:n
        w = -deepcopy(lws[i].values)
        i0 !== nothing && (w .-= w[i0])
        plot!(p2, energies, w; lw=2, color=cols[i])
    end
    ref = Float64.(exact_logdos.values)
    i0 !== nothing && isfinite(ref[i0]) && (ref .-= ref[i0])
    plot!(p2, energies, ref; lw=2, color=:black, label="exact")

    rmse = rmse_exact.(lws)
    p3   = scatter(1:n, rmse; ms=4, color=:steelblue,
                   xlabel="iter", ylabel="RMSE",
                   title="$(title) convergence",
                   legend=false, yscale=:log10)

    return plot(p1, p2, p3; layout=(1,3), size=(980,260), margin=4Plots.mm), rmse
end

# ## Serial MUCA
#
# A single Markov chain accumulates the energy histogram.

sys = Ising([L, L])
init!(sys, :random, rng=Xoshiro(1000))
alg = Multicanonical(Xoshiro(1000), get_centers(exact_logdos))

serial_hists, serial_lws = BinnedObject[], BinnedObject[]
for _ in 1:n_iter
    for _ in 1:(sweeps_therm  * length(sys.spins))
        spin_flip!(sys, alg)
    end
    reset!(alg)
    for _ in 1:(sweeps_record * length(sys.spins))
        ;spin_flip!(sys, alg)
    end
    update!(ensemble(alg); mode=:simple)
    push!(serial_hists, deepcopy(alg.ensemble.histogram))
    push!(serial_lws,   deepcopy(alg.ensemble.logweight))
end

plt, rmse = plot_muca(serial_hists, serial_lws; title="Serial")
println("Serial RMSE = ", round(last(rmse), digits=4))
display(plt)

# ## Production runs
#
# For parallel multicanonical with MPI, see the standalone script:
# ```bash
# mpiexec -n 4 julia --project=docs \
#     docs/src/examples/spin_systems/muca_Ising2D_mpi.jl
# ```
# For parallel multicanonical with threads, see:
# ```bash
# julia --threads=4 --project docs/src/examples/infrastructure/parallel_chains_threads.jl
# ```
