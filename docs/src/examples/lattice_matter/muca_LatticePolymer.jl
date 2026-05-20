# %%                                                #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Multicanonical Sampling of Lattice Polymers
#
# We use multicanonical sampling in the inter-molecular contact energy
# to study the aggregation-disaggregation transition of lattice polymers.
# At low temperature, polymers collapse and aggregate into a compact
# cluster; at high temperature they form dilute coils.  The transition
# is first-order-like for multiple polymers, and the canonical energy
# histogram becomes bimodal.  MUCA flattens this barrier, enabling
# efficient sampling of both phases.

using Random, StatsBase
using MonteCarloX, MCXLatticeMatter
using Plots, ProgressMeter
import MonteCarloX: update!

# ## CI parameters

const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

num_iter             = CI_MODE ? 3       : 20
thermalization_steps = CI_MODE ? 100     : 10_000
recording_steps      = CI_MODE ? 1_000   : 100_000;

# ## System setup
#
# Self-avoiding polymers on a 2D square lattice with periodic boundaries.
# Inter-molecular nearest-neighbor contacts contribute energy ``-J``.

num_poly    = CI_MODE ? 2  : 4
length_poly = CI_MODE ? 8  : 16
L           = CI_MODE ? 20 : 40

sys = LatticePolymer(; dims=[L, L], num_poly=num_poly, length_poly=length_poly,
                       J_intra=0.0, J_inter=1.0)
init!(sys, :random; rng=Xoshiro(42));

# ## Multicanonical ensemble
#
# The observable for MUCA is the total energy, dominated by inter-polymer
# contacts.  We estimate the energy range from the system size.

rng = Xoshiro(42)

E_min = -2 * num_poly * length_poly
E_max = 0
dE    = 1.0

ens = MulticanonicalEnsemble(E_min:dE:E_max)
alg = Metropolis(rng, ens);

# ## Moves
#
# We combine slither (reptation) and translate moves for efficient
# sampling.  Slither changes local conformation while translate
# moves entire polymers.

function sweep!(sys, alg, n_moves)
    for _ in 1:n_moves
        if rand(alg.rng) < 0.7
            slither!(sys, alg)
        else
            translate!(sys, alg)
        end
    end
end

# ## MUCA iteration

E_lo = E_min + 0.5
E_hi = E_max - 0.5
rt   = Roundtrips(Float64(E_lo), Float64(E_hi))

histograms    = Vector{typeof(ens.histogram)}()
logweights    = Vector{typeof(ens.logweight)}()
acceptrate    = Float64[]
flatness_log  = Float64[]
roundtrip_log = Int[]

@showprogress 1 "Iterating MUCA..." for iter in 1:num_iter
    ## thermalization
    sweep!(sys, alg, thermalization_steps)
    ## recording
    reset!(alg)
    fill!(ens.histogram.values, 0.0)
    reset!(rt)
    for _ in 1:recording_steps
        sweep!(sys, alg, 1)
        update!(rt, Float64(energy(sys)))
    end
    ## update weights
    update!(ens; mode=:recursive)
    smooth!(ens, (E_min, E_max); window=3)
    ## log diagnostics
    push!(histograms, deepcopy(ens.histogram))
    push!(logweights, deepcopy(ens.logweight))
    push!(acceptrate, acceptance_rate(alg))
    push!(flatness_log, flatness(ens.histogram, E_min, E_max))
    push!(roundtrip_log, rt.count)
end

# ## Convergence

p1 = plot(acceptrate; xlabel="Iteration", ylabel="Acceptance rate",
          label=nothing, ylims=(0,1))
p2 = plot(flatness_log; xlabel="Iteration", ylabel="Flatness (max/mean)",
          label=nothing, yscale=:log10)
hline!(p2, [2.0]; ls=:dash, color=:gray, label="threshold")
p3 = plot(roundtrip_log; xlabel="Iteration", ylabel="Round trips",
          label=nothing)
plot(p1, p2, p3; layout=(1,3), size=(980, 260), margin=4Plots.mm)

# ## Histograms and log-weights
#
# Converged iterations show a flat energy histogram spanning from the
# disaggregated coil state (``E \approx 0``) to the aggregated cluster
# (``E \ll 0``).

function plot_histograms_and_logweights(xlabel, hist_vec, lw_vec; title_prefix="")
    n    = length(hist_vec)
    cols = palette(:viridis, max(n, 2))[1:n]
    xs   = get_centers(hist_vec[1])

    p1 = plot(xs, hist_vec[1].values; lw=2, color=cols[1],
              xlabel=xlabel, ylabel="counts",
              title="$(title_prefix) histograms", legend=false)
    for i in 2:n
        plot!(p1, xs, hist_vec[i].values; lw=2, color=cols[i])
    end

    p2 = plot(xlabel=xlabel, ylabel="-log w",
              title="$(title_prefix) log-weights", legend=false)
    for i in 1:n
        plot!(p2, xs, -lw_vec[i].values .+ lw_vec[i].values[1]; lw=2, color=cols[i])
    end

    return plot(p1, p2; layout=(@layout([a b])), size=(960, 320), margin=4Plots.mm)
end

plot_histograms_and_logweights("E", histograms, logweights; title_prefix="Lattice polymer MUCA")
