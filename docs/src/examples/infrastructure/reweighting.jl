# %%                                                #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Reweighting
#
# Reweighting is importance sampling applied *after the fact*: a chain records a
# coordinate under one ensemble, and we ask for expectations under another
# without rerunning. Each sample carries a log weight
# ``g_i = \log w_\text{target}(x_i) - \log w_\text{source}(x_i)``; [`reweight`](@ref)
# forms these in log space and returns an `ImportanceWeights` whose `weights`,
# `ess`, and `log_normalization` plug into StatsBase.
#
# The coordinate here is the energy of a 2D Ising model. The full simulation lives
# in `examples/reweighting.jl`; this page loads its recorded output and shows the
# reweighting calls themselves — three targets, all validated against the exact
# density of states ([Beale 1996](https://doi.org/10.1103/PhysRevLett.76.78)).

using MonteCarloX, StatsBase

using DelimitedFiles, Plots                                                            # hide
N = 8 * 8                                                                              # hide
function reweight_data(name)                                                           # hide
    p1 = joinpath(@__DIR__, "..", "..", "data", name)                                  # hide
    p2 = joinpath(@__DIR__, "..", "data", name)                                        # hide
    isfile(p1) || isfile(p2) || error("run: julia --project=examples examples/reweighting.jl docs/src/data")  # hide
    return readdlm(isfile(p1) ? p1 : p2, '\t'; header = true)[1]                       # hide
end                                                                                    # hide
nothing                                                                                # hide

# ## 1. Single-histogram reweighting (Metropolis)
#
# A canonical chain at ``\beta_0 = 0.3`` recorded its energies. Its samples *are*
# the canonical distribution at ``\beta_0``, so reweighting to a nearby ``\beta``
# only rescales each weight by ``e^{-(\beta-\beta_0)E}``:

energies = reweight_data("reweighting_energies_canonical.tsv")[:, 1]                   # hide
source   = BoltzmannEnsemble(β = 0.30)                        # the chain's own ensemble
iw       = reweight(energies, source, BoltzmannEnsemble(β = 0.32))
(; avgE = mean(energies, weights(iw)) / N, ess = round(Int, ess(iw)))

# But as ``\beta`` moves away from ``\beta_0``, the target's important energies sit
# in the tails of what was sampled, the weights concentrate on a few draws, and the
# effective sample size collapses — `ess` makes that failure visible instead of
# silent. Reweighting the full run across a window shows exactly that:

exact = reweight_data("reweighting_exact_avgE.tsv")                                    # hide
sh    = reweight_data("reweighting_single_histogram.tsv")                              # hide
p1 = plot(exact[:, 1], exact[:, 2]; lw = 3, color = :black, label = "exact",          # hide
          xlabel = "β", ylabel = "⟨E⟩/N", title = "single-histogram reweight")        # hide
scatter!(p1, sh[:, 1], sh[:, 2]; ms = 3, color = 1, label = "reweighted")             # hide
p2 = plot(sh[:, 1], sh[:, 3]; lw = 2, color = 2, legend = false,                       # hide
          xlabel = "β", ylabel = "ESS / N", title = "effective sample size")          # hide
plot(p1, p2; layout = (1, 2), size = (900, 320), margin = 4Plots.mm)                  # hide

# The reweighted averages track the exact curve only in a narrow band around
# ``\beta_0``. To cover a wide temperature range from a *single* run we need a
# source that samples all energies.

# ## 2. Multicanonical reweighting to any temperature
#
# A multicanonical run flattens the energy histogram, so its samples span the whole
# spectrum. Only the `source` changes — it is now the converged multicanonical
# ensemble — and the *same* call reinstates ``e^{-\beta E}`` at *any* ``\beta`` we ask for:

energies_muca = reweight_data("reweighting_energies_muca.tsv")[:, 1]                   # hide
mw   = reweight_data("reweighting_muca_logweight.tsv")                                 # hide
W    = Dict(mw[i, 1] => mw[i, 2] for i in axes(mw, 1))                                 # hide
muca = FunctionEnsemble(E -> W[E + 0.0]) # converged multicanonical log-weights W(E)  # hide
iw   = reweight(energies_muca, muca, BoltzmannEnsemble(β = 0.44))
mean(energies_muca, weights(iw)) / N

# One run yields the whole ``\langle E\rangle_\beta`` curve, across the transition
# where the single-histogram method broke down:

muca_curve = reweight_data("reweighting_muca_avgE.tsv")                                # hide
plt = plot(exact[:, 1], exact[:, 2]; lw = 3, color = :black, label = "exact",         # hide
           xlabel = "β", ylabel = "⟨E⟩/N", title = "multicanonical → any β",          # hide
           legend = :topright)                                                        # hide
scatter!(plt, muca_curve[:, 1], muca_curve[:, 2]; ms = 3, color = 3, label = "muca reweighted")  # hide
plt                                                                                    # hide

# ## 3. Multicanonical reweighting to the density of states
#
# The flat target is the special case where reweighting removes *all* weighting, so
# it is the default — no target argument. Multicanonical samples are distributed as
# ``g(E)\,e^{W(E)}``; reweighting by ``e^{-W(E)}`` makes a weighted histogram of the
# energies proportional to ``g(E)`` itself:

iw = reweight(energies_muca, muca)       # flat (constant) target ⇒ density of states
round(ess(iw); digits = 1)

# A weighted histogram of the energies under `weights(iw)` then reconstructs
# ``g(E)`` across roughly a hundred orders of magnitude:

dos = reweight_data("reweighting_dos.tsv")                                            # hide
plt = plot(dos[:, 1], dos[:, 3]; lw = 3, color = :black, label = "exact log g(E)",    # hide
           xlabel = "E", ylabel = "log g(E)", title = "multicanonical → density of states",  # hide
           legend = :bottom)                                                          # hide
scatter!(plt, dos[:, 1], dos[:, 2]; ms = 3, color = 4, label = "reweighted")          # hide
plt                                                                                    # hide

# The same converged run delivered every canonical temperature *and* the density of
# states, from one set of recorded energies — the reweighting call is the only thing
# that changed.
