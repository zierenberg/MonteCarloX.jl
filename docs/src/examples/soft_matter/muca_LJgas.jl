# %%                                                #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
# include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Multicanonical Sampling of a Lennard-Jones Gas
#
# We use multicanonical sampling in the total energy to drive a
# dilute Lennard-Jones gas through the condensation-evaporation
# transition. At subcritical temperatures and low density, the
# system exhibits a first-order-like droplet transition: the energy
# histogram in a canonical simulation is bimodal with a suppressed
# barrier region. MUCA flattens this histogram, enabling efficient
# sampling of both the gas and droplet phases.
#
# Reference: Zierenberg & Janke, Phys. Rev. E (2015)

using Random, StatsBase
using MonteCarloX, MCXSoftMatter
using Plots, ProgressMeter
import MonteCarloX: update!
import MCXSoftMatter: translate!

# ## CI parameters

const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

num_iter       = CI_MODE ? 3     : 20;
sweeps_equil   = CI_MODE ? 10    : 10_000;
sweeps_measure = CI_MODE ? 100   : 1_000_000;

# ## System setup
#
# A 3D Lennard-Jones gas with truncated-shifted potential at
# ``r_{\rm cut} = 2.5\sigma``.  We work at subcritical temperature
# ``T = 0.7\,\varepsilon/k_B`` and low density ``\rho = 0.05\sigma^{-3}``
# where the condensation transition is accessible for small ``N``.
# 
# can be visualized with scatter3d(eachrow(reduce(hcat, sys.positions))...)

N   = CI_MODE ? 20  : 16
rho = 0.01
e_min = -3.3 # prior knowledge
e_max =  0.0
T_max =  1.0
T_min =  0.5

r0 = 1.0
sigma = r0 / 2^(1/6)
lj  = LennardJonesPotential(epsilon=1.0, sigma=sigma, r_cutoff=2.5*sigma)
sys = ParticleGas(; N=N, rho=rho, pair_potential=lj)
init!(sys, :random; rng=Xoshiro(42));

dx_short = 0.1;
dx_long = mean(sys.env.L) ./ 4;
@inline function sweep!(sys, alg)
    for i in 1:num_particles(sys)
        translate!(sys, alg, dx_short)
    end
    translate!(sys, alg, dx_long)
    return nothing
end;

# ## Multicanonical ensemble
#
# We set up the MUCA ensemble over the energy range. For a dilute gas
# the energy is mostly near zero; the condensed phase has large negative
# energy.  We estimate bounds from a short canonical pre-run.

rng = Xoshiro(42)

E_min = e_min * N
E_max = e_max * N
dE    = CI_MODE ? 2.0 : 0.1

# add a buffer to the desired range to allow for manual supression at the boundaries
alg = Multicanonical(rng, E_min - N:dE:E_max + N; warn_overwrite=false);
ens = ensemble(alg);
rt  = Roundtrips(E_min, E_max);

# ## MUCA iteration
#
# Each iteration thermalizes, records the energy histogram, and updates
# the multicanonical weights using the recursive scheme.  We also
# monitor histogram flatness and round trips through the energy range.

set!(ensemble(alg), E -> - E/T_max)

histograms   = Vector{typeof(ens.histogram)}()
logweights   = Vector{typeof(ens.logweight)}()
acceptrate   = Float64[]
flatness_log = Float64[]
roundtrip_log = Int[]

@showprogress 1 "Iterating MUCA..." for iter in 1:num_iter
    ## thermalization
    for _ in 1:sweeps_equil; sweep!(sys,alg); end
    ## recording
    reset!(alg)
    reset!(rt)
    for _ in 1:sweeps_measure/num_iter*iter
        sweep!(sys, alg)
        update!(rt, energy(sys))
    end
    # update weights
    update!(ens; mode=:recursive)
    # help the walker move into low-energy regions by extrapolating from the first non-empty bins at the boundaries
    # E_low, E_high = visited_range(ens)
    # extend!(ensemble(alg), :low; anchor=E_low, slope=-1/T_min)
    # manual suppression at boundaries by applying temperature ramp outside desired range
    extend!(ensemble(alg), :high; anchor=E_max, slope=-1/T_max)
    extend!(ensemble(alg), :low;  anchor=E_min, slope=-1/T_min)
    # smooth!(ens, (E_min, E_max); window=3)
    # log diagnostics
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
# dilute gas (``E \approx 0``) to the condensed droplet
# (``E \ll 0``).

function plot_histograms_and_logweights(xlabel, hist_vec, lw_vec; title_prefix="")
    n    = length(hist_vec)
    cols = palette(:viridis, max(n, 2))[1:n]
    xs   = get_centers(hist_vec[1]) ./ N

    p1 = plot(xs, hist_vec[1].values; lw=2, color=cols[1],
              xlabel=xlabel, ylabel="counts",
              title="$(title_prefix)", legend=false)
    for i in 2:n
        plot!(p1, xs, hist_vec[i].values; lw=2, color=cols[i])
    end

    p2 = plot(xlabel=xlabel, ylabel="log w",
              title="$(title_prefix)", legend=false)
    for i in 1:n
        plot!(p2, xs, lw_vec[i].values .- lw_vec[i].values[1]; lw=2, color=cols[i])
    end

    return plot(p1, p2; layout=(@layout([a b])), size=(960, 320), margin=4Plots.mm)
end

plot_histograms_and_logweights("E/N", histograms, logweights; title_prefix="LJ gas MUCA (N=$(N))")