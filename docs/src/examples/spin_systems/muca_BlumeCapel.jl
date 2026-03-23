# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Multicanonical Sampling of the Blume-Capel Model
#
# Illustration of multicanonical sampling in only part of the Hamiltonian.
# Specifically, the Hamiltonian of the Blume-Capel model reads
# ```math
#    H = -J\sum_{ij}s_i s_j + \Delta\sum_i s_i^2
# ```
# The spin-spin interaction remains in the canonical (Boltzmann) ensemble
# while we construct a multicanonical ensemble for the crystal-field term.
# Depending on the temperature, a change in ``\Delta`` induces no transition,
# a second-order, or a first-order phase transition.

using Random, StatsBase
using MonteCarloX, SpinSystems
using Plots, ProgressMeter

# ## CI parameters

const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

num_iter           = CI_MODE ? 3       : 20
thermalization_steps = CI_MODE ? 100   : 10_000
recording_steps    = CI_MODE ? 1_000   : 1_000_000;

# ## Parameters

L   = 8
T   = 0.9
sys = BlumeCapel([L, L]);

# ## Custom ensemble
#
# We combine a Boltzmann weight for the pairwise interaction with a
# multicanonical weight for the crystal-field term ``\sum_i s_i^2``.
# The `CustomEnsemble` routes each contribution to the appropriate ensemble.

mutable struct CustomEnsemble <: AbstractEnsemble
    pair          :: BoltzmannEnsemble
    spin2         :: MulticanonicalEnsemble
    record_visits :: Bool
end

@inline function MonteCarloX.logweight(lw::CustomEnsemble, H::Tuple{<:Real,<:Real})
    return MonteCarloX.logweight(lw.pair, H[1]) + MonteCarloX.logweight(lw.spin2, H[2])
end

@inline MonteCarloX.should_record_visit(lw::CustomEnsemble) = lw.record_visits

@inline function MonteCarloX.record_visit!(lw::CustomEnsemble, H_vis::Tuple{<:Real,<:Real})
    MonteCarloX.record_visit!(lw.spin2, H_vis[2])
    return nothing
end

@inline function reset_histogram!(lw::CustomEnsemble)
    fill!(lw.spin2.histogram.values, zero(eltype(lw.spin2.histogram.values)))
    return nothing
end

@inline reset_histogram!(alg::AbstractImportanceSampling) =
    reset_histogram!(MonteCarloX.ensemble(alg))

ens = CustomEnsemble(
    BoltzmannEnsemble(T=T),
    MulticanonicalEnsemble(0:1:length(sys.spins)),
    true,
);

# ## Spin flip
#
# We implement a custom `spin_flip!` for the Blume-Capel model that evaluates
# the two-component observable ``(J\sum s_i s_j,\, \sum s_i^2)`` and passes
# it to `accept!` as a tuple — the `CustomEnsemble` routes each component
# to the correct acceptance weight.

function spin_flip!(sys::SpinSystems.AbstractBlumeCapel, alg::AbstractImportanceSampling)
    i     = pick_site(alg.rng, length(sys.spins))
    s_new = SpinSystems._propose_state(alg.rng, sys.spins[i])
    Δpair, Δspin, Δspin2 = SpinSystems.propose_changes(sys, i, s_new)
    H_old = (sys.J * sys.sum_pair_interactions, sys.sum_spins2)
    H_new = (H_old[1] + sys.J * Δpair, H_old[2] + Δspin2)
    accept!(alg, H_new, H_old) && modify!(sys, i, s_new, Δpair, Δspin, Δspin2)
    return nothing
end

# ## Multicanonical iteration
#
# Each iteration thermalizes the system, resets the histogram, accumulates
# statistics, and updates the multicanonical weights via the Wang-Landau rule.

rng = Xoshiro(42)
alg = Metropolis(rng, ens)

histograms = Vector{typeof(ensemble(alg).spin2.histogram)}()
logweights = Vector{typeof(ensemble(alg).spin2.logweight)}()
acceptrate = Float64[]

@showprogress 1 "Iterating MUCA..." for _ in 1:num_iter
    for _ in 1:thermalization_steps; spin_flip!(sys, alg); end
    reset!(alg)
    reset_histogram!(alg)
    for _ in 1:recording_steps;     spin_flip!(sys, alg); end
    MonteCarloX.update!(ensemble(alg).spin2)
    push!(histograms, deepcopy(ensemble(alg).spin2.histogram))
    push!(logweights, deepcopy(ensemble(alg).spin2.logweight))
    push!(acceptrate, acceptance_rate(alg))
end

# ## Convergence
#
# The acceptance rate should stabilise as the weights converge to flat.

plot(acceptrate; xlabel="Iteration", ylabel="Acceptance rate",
     label=nothing, size=(600, 220), margin=5Plots.mm, ylims=(0,1))

# ## Histograms and log-weights
#
# Each iteration refines the estimated log-DOS for ``\sum_i s_i^2``.
# Converged iterations should show a flat histogram and a smooth log-weight.

function plot_histograms_and_logweights(xlabel, hist_vec, lw_vec; title_prefix="")
    n    = length(hist_vec)
    cols = palette(:viridis, max(n, 2))[1:n]
    xs   = get_centers(hist_vec[1])
    i0   = 1

    p1 = plot(xs, hist_vec[1].values; lw=2, color=cols[1],
              xlabel=xlabel, ylabel="counts",
              title="$(title_prefix) histograms", legend=false,
              ylims=(0, maximum(hist_vec[1].values) * 1.2))
    for i in 2:n
        plot!(p1, xs, hist_vec[i].values; lw=2, color=cols[i])
    end

    p2 = plot(xlabel=xlabel, ylabel="-log w",
              title="$(title_prefix) log-weights", legend=false)
    for i in 1:n
        plot!(p2, xs, -lw_vec[i].values .+ lw_vec[i].values[i0]; lw=2, color=cols[i])
    end

    return plot(p1, p2; layout=(@layout([a b])), size=(960, 320), margin=4Plots.mm)
end

plot_histograms_and_logweights("∑sᵢ²", histograms, logweights; title_prefix="MUCA")