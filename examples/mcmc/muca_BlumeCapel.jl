# # Multicanonical Sampling of the Blume-Capel Model
#
# Illustration of multicanonical sampling in only part of the Hamiltonian
# [Zierenberg, Fytas & Janke 2015].
# Specifically, the Hamiltonian of the Blume-Capel model reads
# ```math
#    H = -J\sum_{ij}s_i s_j + \Delta\sum_i s_i^2
# ```
# The spin-spin interaction remains in the canonical (Boltzmann) ensemble
# while we construct a multicanonical ensemble for the crystal-field term.
# Depending on the temperature, a change in ``\Delta`` induces no transition,
# a second-order, or a first-order phase transition.

using Random, StatsBase, Plots, DelimitedFiles
using MonteCarloX, MCXSpins

datadir   = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "data")))  # hide
hist_file = joinpath(datadir, "muca_BlumeCapel_histogram.tsv")     # hide
lw_file   = joinpath(datadir, "muca_BlumeCapel_logweight.tsv")     # hide
diag_file = joinpath(datadir, "muca_BlumeCapel_diagnostics.tsv")   # hide

L                     = 8
T                     = 0.9
num_iter              = 20
thermalization_sweeps = 200        # one sweep = one attempted flip per site
recording_sweeps      = 16_000
nothing # hide

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

@inline reset_histogram!(alg::AbstractMarkovChainMonteCarlo) =
    reset_histogram!(MonteCarloX.ensemble(alg))
nothing # hide

# ## Spin flip
#
# We implement a custom `spin_flip!` for the Blume-Capel system that evaluates
# the two-component observable ``(\sum s_i s_j,\, \sum s_i^2)`` and passes
# it to `accept!` as a tuple — the `CustomEnsemble` routes each component
# to the correct acceptance weight. Both components are exactly the interaction
# caches (the coupling-free sums the pair and crystal-field terms maintain), and
# their changes are the first two entries of the delta payload tuple.

function spin_flip!(sys::SpinSystem{<:Any, <:Spin{1//1}}, alg::MetropolisHastingsAlgorithm)
    i     = pick_site(alg.rng, length(sys.spins))
    s_new = propose_state(alg.rng, sys, i)
    δs    = MCXSpins.delta_sys(sys, i, s_new)
    pair, crystal = sys.interactions
    H_old = (pair.cache.val, crystal.cache.val)
    H_new = (H_old[1] + δs[1], H_old[2] + δs[2])
    accept!(alg, H_new, H_old) && modify!(sys, i, s_new, δs)
    return nothing
end
nothing #hide

# ## Multicanonical iteration
#
# Each iteration thermalizes, records a histogram, and updates weights using the
# recursive scheme (Berg, 1996; Janke, 1998), which accumulates statistics across
# iterations for more stable convergence than the trivial ``W -= \log H`` rule. We
# smooth the weights, and track acceptance, histogram flatness, and round trips
# through the observable range — keeping every iteration to watch convergence.

# One sweep is one attempted flip per site; the function barriers keep the hot loops
# specialized. The recording variant tracks Σ s² round trips through the crystal-field
# cache after every flip.

sweep!(sys, alg, n_sweeps) =
    (for _ in 1:n_sweeps, _ in 1:length(sys.spins); spin_flip!(sys, alg); end)
function record_sweeps!(sys, alg, rt, n_sweeps)
    for _ in 1:n_sweeps, _ in 1:length(sys.spins)
        spin_flip!(sys, alg)
        update!(rt, sys.interactions[2].cache.val)      ## Σ s² from the crystal-field cache
    end
end

if !isfile(lw_file)                                                # hide
sys = BlumeCapelSystem([L, L])
ens = CustomEnsemble(BoltzmannEnsemble(T = T),
                     MulticanonicalEnsemble(0:1:length(sys.spins)), true)
rng = Xoshiro(42)
alg = MetropolisHastingsAlgorithm(rng, ens)

N_sites = length(sys.spins)
s2_min, s2_max = 0, N_sites
rt = Roundtrips(s2_min + 0.01 * N_sites, s2_max - 0.01 * N_sites)

s2 = collect(get_centers(ensemble(alg).spin2.histogram))
H  = zeros(length(s2), num_iter)      # recorded histogram of each iteration
W  = zeros(length(s2), num_iter)      # log-weights after each iteration
acceptrate    = zeros(num_iter)
flatness_log  = zeros(num_iter)
roundtrip_log = zeros(num_iter)

for iter in 1:num_iter
    sweep!(sys, alg, thermalization_sweeps)
    reset!(alg)
    reset_histogram!(alg)
    reset!(rt)
    record_sweeps!(sys, alg, rt, recording_sweeps)
    muca_ens = ensemble(alg).spin2
    update_logweight!(muca_ens; mode = :recursive)
    smooth!(muca_ens, (s2_min, s2_max); window = 3)
    H[:, iter]       = muca_ens.histogram.values
    W[:, iter]       = muca_ens.logweight.values
    acceptrate[iter]    = acceptance_rate(alg)
    flatness_log[iter]  = flatness(muca_ens.histogram, s2_min, s2_max)
    roundtrip_log[iter] = rt.count
end

header = permutedims(["s2"; ["iter$(it)" for it in 1:num_iter]])   # hide
mkpath(datadir)                                                    # hide
writedlm(hist_file, [header; hcat(s2, H)], '\t')                   # hide
writedlm(lw_file,   [header; hcat(s2, W)], '\t')                   # hide
writedlm(diag_file, ["iter" "acceptrate" "flatness" "roundtrips";  # hide
                     hcat(1:num_iter, acceptrate, flatness_log, roundtrip_log)], '\t')  # hide
end                                                                # hide
hh = readdlm(hist_file, '\t'; header = true)[1]                    # hide
ll = readdlm(lw_file,   '\t'; header = true)[1]                    # hide
dd = readdlm(diag_file, '\t'; header = true)[1]                    # hide
s2 = hh[:, 1]; H = hh[:, 2:end]; W = ll[:, 2:end]                  # hide
acceptrate = dd[:, 2]; flatness_log = dd[:, 3]; roundtrip_log = dd[:, 4]  # hide
num_iter = size(H, 2)                                              # hide
nothing # hide

# ## Convergence
#
# We track the acceptance rate, histogram flatness (`max/mean`, → 1 when flat),
# and the number of round trips through the observable range.

p1 = plot(acceptrate; xlabel = "iteration", ylabel = "acceptance rate",
          label = nothing, ylims = (0, 1))
p2 = plot(flatness_log; xlabel = "iteration", ylabel = "flatness (max/mean)",
          label = nothing, yscale = :log10)
hline!(p2, [2.0]; ls = :dash, color = :gray, label = "threshold")
p3 = plot(roundtrip_log; xlabel = "iteration", ylabel = "round trips", label = nothing)
plot(p1, p2, p3; layout = (1, 3), size = (980, 260), margin = 4Plots.mm)

# ## Histograms and log-weights
#
# Each iteration refines the estimated log-DOS for ``\sum_i s_i^2``. Converged
# iterations show a flat histogram (left) and a smooth log-weight (right).

cols = palette(:viridis, max(num_iter, 2))

ph = plot(xlabel = "∑ sᵢ²", ylabel = "counts", title = "MUCA histograms",
          legend = false, ylims = (0, maximum(H[:, 1]) * 1.2))
for it in 1:num_iter
    plot!(ph, s2, H[:, it]; lw = 2, color = cols[it])
end

pw = plot(xlabel = "∑ sᵢ²", ylabel = "-log w", title = "MUCA log-weights", legend = false)
for it in 1:num_iter
    plot!(pw, s2, -W[:, it] .+ W[1, it]; lw = 2, color = cols[it])
end

plot(ph, pw; layout = (1, 2), size = (960, 320), margin = 4Plots.mm)

# ## References
#
# - J. Zierenberg, N. G. Fytas, W. Janke, *Parallel multicanonical study of the three-dimensional
#   Blume-Capel model*, Phys. Rev. E **91**, 032126 (2015).
#   [doi:10.1103/PhysRevE.91.032126](https://doi.org/10.1103/PhysRevE.91.032126)
# - M. Blume, *Theory of the first-order magnetic phase change in UO₂*, Phys. Rev. **141**, 517 (1966).
#   [doi:10.1103/PhysRev.141.517](https://doi.org/10.1103/PhysRev.141.517)
# - H. W. Capel, *On the possibility of first-order phase transitions in Ising systems of triplet
#   ions with zero-field splitting*, Physica **32**, 966 (1966).
#   [doi:10.1016/0031-8914(66)90027-9](https://doi.org/10.1016/0031-8914(66)90027-9)
