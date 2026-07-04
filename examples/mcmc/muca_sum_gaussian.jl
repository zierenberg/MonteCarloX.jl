# # Large Deviation Theory: Sum of Gaussian Random Variables
#
# Large deviation theory (LDT) describes the probability of observing rare,
# atypical fluctuations of macroscopic observables far from their mean. For a sum
# ``S_N = \sum_{i=1}^N X_i`` of i.i.d. random variables the central limit theorem
# governs typical fluctuations, but LDT governs the exponentially rare tails.
#
# This example uses the sum of ``N`` Gaussian variables as a tractable benchmark —
# the exact distribution is known analytically — and compares three sampling
# strategies: direct sampling, importance sampling with a biasing field ``\beta``,
# and multicanonical sampling (flat histogram across the full distribution).

using Random, Distributions, StatsBase, LinearAlgebra, DelimitedFiles
using MonteCarloX
using Plots

datadir     = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "data")))  # hide
dist_file   = joinpath(datadir, "muca_sum_gaussian_distributions.tsv")  # hide
iterh_file  = joinpath(datadir, "muca_sum_gaussian_iter_hist.tsv")      # hide
iterw_file  = joinpath(datadir, "muca_sum_gaussian_iter_logweight.tsv") # hide
accept_file = joinpath(datadir, "muca_sum_gaussian_acceptance.tsv")     # hide

n_direct     = 100_000
n_therm      = 10_000
n_metro      = 1_000_000
n_muca       = 10_000_000
n_muca_known = 50_000_000
n_iter       = 10
n_iter_steps = 5_000_000

μ, σ, N = 0.0, 1.0, 100

## exact distribution of S_N is Normal(N μ, √N σ)
pdf_sum     = Normal(N * μ, sqrt(N) * σ)
std_sum     = sqrt(N) * σ
bins_sum    = (N * μ - 10 * std_sum):(std_sum / 10):(N * μ + 10 * std_sum)
centers_sum = collect(bins_sum[1:end-1] .+ diff(collect(bins_sum)) ./ 2)
logpdf_sum  = logpdf.(pdf_sum, centers_sum)
βs          = [-0.5, 0.0, 0.5]
density(w)  = w ./ (sum(w) * step(bins_sum))
nothing # hide

# ## System definition
#
# The state is a vector of ``N`` Gaussian variables together with their running
# sum. Only the sum enters the sampling weight; each variable evolves under its own
# Gaussian prior via a Metropolis-within-Gibbs step.

mutable struct SumGaussianRVs
    logpdf_rv :: Function
    rvs       :: Vector{Float64}
    sum_rvs   :: Float64
    function SumGaussianRVs(rng, N; μ = 0.0, σ = 1.0)
        dist = Normal(μ, σ)
        rvs  = rand(rng, dist, N)
        new(x -> logpdf(dist, x), rvs, sum(rvs))
    end
end

sum_rvs(sys::SumGaussianRVs) = sys.sum_rvs

# Each update proposes a move for one randomly chosen variable, filtered first by
# the Gaussian prior (Metropolis-within-Gibbs) and then accepted by the sampling
# algorithm based on the sum.

function update!(sys::SumGaussianRVs, alg::AbstractMarkovChainMonteCarlo; δ = 0.5)
    idx    = rand(alg.rng, 1:length(sys.rvs))
    rv_old = sys.rvs[idx]
    rv_new = rv_old + δ * (2 * rand(alg.rng) - 1)
    if rand(alg.rng) < exp(sys.logpdf_rv(rv_new) - sys.logpdf_rv(rv_old))
        sum_new = sys.sum_rvs + (rv_new - rv_old)
        if accept!(alg, sum_new, sys.sum_rvs)
            sys.sum_rvs  = sum_new
            sys.rvs[idx] = rv_new
        end
    else
        alg.steps += 1
    end
end

# ## Running the samplers
#
# All strategies below are expensive (up to ``5\times10^7`` steps), so we run them
# once and cache the resulting distributions; a docs build only reloads them.
# **Direct sampling** draws independent sums as a reference. **Importance sampling**
# tilts the target by ``e^{\beta S_N}`` to reach the tails. **Multicanonical**
# sampling learns flat weights so the whole support — including rare tails — is
# visited uniformly, either from scratch or from known/iterated weights.

if !isfile(dist_file)                                          # hide
## direct sampling
direct_samples = [sum(μ .+ σ .* randn(MersenneTwister(i), N)) for i in 1:n_direct]
dist_direct    = normalize(fit(Histogram, direct_samples, bins_sum); mode = :pdf).weights

## importance sampling at several β
function run_metropolis(β)
    sys  = SumGaussianRVs(MersenneTwister(23), N; μ = μ, σ = σ)
    alg  = Metropolis(MersenneTwister(42); β = β)
    meas = Measurements([:sum => sum_rvs => fit(Histogram, [], bins_sum)], interval = 1)
    for _ in 1:n_therm; update!(sys, alg); end
    for i in 1:n_metro; update!(sys, alg); measure!(meas, sys, i); end
    return density(meas[:sum].data.weights)
end
is_dists = [run_metropolis(β) for β in βs]

## multicanonical with flat weights
sys_muca0 = SumGaussianRVs(MersenneTwister(23), N; μ = μ, σ = σ)
alg_muca0 = Multicanonical(MersenneTwister(100), bins_sum; init = 0.0)
for _ in 1:n_therm; update!(sys_muca0, alg_muca0); end
reset!(alg_muca0)
for _ in 1:n_muca; update!(sys_muca0, alg_muca0); end
dist_flat = density(alg_muca0.ensemble.histogram.values)

## multicanonical initialised with the known log-PDF plus linear boundary tails
sys_known = SumGaussianRVs(MersenneTwister(42), N; μ = μ, σ = σ)
alg_known = Multicanonical(MersenneTwister(42), bins_sum)
lw        = logweight(alg_known)
cs        = get_centers(lw)
set!(lw, (first(cs), last(cs)), x -> -logpdf(pdf_sum, x))
x_left, x_right = -3 * std_sum, +3 * std_sum
set!(lw, (first(cs), x_left),  x -> lw(x_left)  + (x - x_left)  * 2.0)
set!(lw, (x_right,  last(cs)), x -> lw(x_right) - (x - x_right) * 2.0)
lw_init = copy(get_values(lw))
for _ in 1:n_therm; update!(sys_known, alg_known; δ = 0.1); end
reset!(alg_known)
for _ in 1:n_muca_known; update!(sys_known, alg_known; δ = 0.1); end
dist_known = density(alg_known.ensemble.histogram.values)

## multicanonical iteration from flat weights
sys_iter = SumGaussianRVs(MersenneTwister(42), N; μ = μ, σ = σ)
alg_iter = Multicanonical(MersenneTwister(42), bins_sum)
lw_iter  = logweight(alg_iter)
cs_iter  = get_centers(lw_iter)
i_left   = searchsortedfirst(cs_iter, first(bins_sum) + std_sum)
i_right  = searchsortedlast(cs_iter,  last(bins_sum)  - std_sum)
iter_hist   = zeros(length(centers_sum), n_iter)
iter_lw     = zeros(length(centers_sum), n_iter)
iter_accept = zeros(n_iter)
for it in 1:n_iter
    for _ in 1:n_therm;      update!(sys_iter, alg_iter); end
    reset!(alg_iter)
    for _ in 1:n_iter_steps; update!(sys_iter, alg_iter); end
    MonteCarloX.update!(ensemble(alg_iter); mode = :simple)
    wl = get_values(lw_iter)[i_left]; wr = get_values(lw_iter)[i_right]
    set!(lw_iter, (first(cs_iter), cs_iter[i_left]), x -> wl + (x - cs_iter[i_left]) * 2.0)
    set!(lw_iter, (cs_iter[i_right], last(cs_iter)), x -> wr - (x - cs_iter[i_right]) * 2.0)
    iter_hist[:, it]  = ensemble(alg_iter).histogram.values
    iter_lw[:, it]    = ensemble(alg_iter).logweight.values
    iter_accept[it]   = acceptance_rate(alg_iter)
end

dheader = permutedims(["S_N", "direct", "is_m0.5", "is_0.0", "is_0.5",   # hide
                       "muca_flat", "lw_init", "muca_known"])            # hide
mkpath(datadir)                                                          # hide
writedlm(dist_file, [dheader;                                            # hide
    hcat(centers_sum, dist_direct, is_dists[1], is_dists[2], is_dists[3], # hide
         dist_flat, lw_init, dist_known)], '\t')                        # hide
iheader = permutedims(["S_N"; ["iter$(it)" for it in 1:n_iter]])         # hide
writedlm(iterh_file, [iheader; hcat(centers_sum, iter_hist)], '\t')      # hide
writedlm(iterw_file, [iheader; hcat(centers_sum, iter_lw)], '\t')        # hide
writedlm(accept_file, ["iter" "acceptance"; hcat(1:n_iter, iter_accept)], '\t')  # hide
end                                                                       # hide
dd = readdlm(dist_file, '\t'; header = true)[1]                          # hide
ih = readdlm(iterh_file, '\t'; header = true)[1]                         # hide
iw = readdlm(iterw_file, '\t'; header = true)[1]                         # hide
ac = readdlm(accept_file, '\t'; header = true)[1]                        # hide
dist_direct = dd[:, 2]; is_dists = [dd[:, 3], dd[:, 4], dd[:, 5]]        # hide
dist_flat = dd[:, 6]; lw_init = dd[:, 7]; dist_known = dd[:, 8]          # hide
iter_hist = ih[:, 2:end]; iter_lw = iw[:, 2:end]; iter_accept = ac[:, 2] # hide
n_iter = size(iter_hist, 2)                                              # hide
nothing # hide

# ## Direct sampling
#
# Independent draws reproduce the bulk of the distribution but never reach the tails.

plot(centers_sum, dist_direct; st = :bar, linewidth = 0, alpha = 0.6, label = "direct",
     xlabel = "Sₙ", ylabel = "density", title = "direct sampling vs exact",
     size = (600, 250), margin = 5Plots.mm)
plot!(centers_sum, pdf.(pdf_sum, centers_sum); lw = 2, color = :black, ls = :dash, label = "exact")

# ## Importance sampling
#
# A biasing field ``\beta`` tilts sampling toward larger (``\beta>0``) or smaller
# (``\beta<0``) sums; together the biased runs cover the full distribution.

p_is = plot(centers_sum, pdf.(pdf_sum, centers_sum); lw = 2, color = :black, ls = :dash,
            label = "exact", xlabel = "Sₙ", ylabel = "density", title = "importance sampling",
            size = (600, 250), margin = 5Plots.mm)
for (k, β) in enumerate(βs)
    plot!(p_is, centers_sum, is_dists[k]; lw = 2, label = "β = $β")
end
p_is

# ## Multicanonical — flat weights
#
# With ``w = 1`` the multicanonical algorithm samples the full distribution
# uniformly — including the rare tails — without knowing the target in advance.

plot(centers_sum, dist_flat; lw = 2, label = "MUCA (flat weights)",
     xlabel = "Sₙ", ylabel = "density", title = "multicanonical flat histogram",
     size = (600, 250), margin = 5Plots.mm)
plot!(centers_sum, pdf.(pdf_sum, centers_sum); lw = 2, color = :black, ls = :dash, label = "exact")

# ## Multicanonical — known weights
#
# If the target is known, we initialise the weights to the exact ``-\log`` PDF plus
# linear boundary tails. The initialised log-weight already matches the exact curve.

plot(centers_sum, lw_init; lw = 2, label = "initialised log-weight",
     xlabel = "Sₙ", ylabel = "log w(Sₙ)", title = "log-weight vs exact −log PDF",
     size = (600, 250), margin = 5Plots.mm)
plot!(centers_sum, -logpdf_sum; lw = 2, color = :black, ls = :dash, label = "exact −log PDF")

# Sampling under those weights recovers the exact density across the full support.

plot(centers_sum, dist_known; lw = 2, label = "MUCA (known weights)",
     xlabel = "Sₙ", ylabel = "density", title = "MUCA with exact initialisation",
     size = (600, 250), margin = 5Plots.mm)
plot!(centers_sum, pdf.(pdf_sum, centers_sum); lw = 2, color = :black, ls = :dash, label = "exact")

# ## Multicanonical iteration
#
# Starting from flat weights, the algorithm iteratively refines the log-weights
# until the histogram is flat; the estimated log-DOS converges onto the exact
# reference and the acceptance rate stabilises.

plot(iter_accept; xlabel = "iteration", ylabel = "acceptance rate",
     title = "convergence of acceptance rate", label = nothing,
     size = (600, 220), margin = 5Plots.mm)

# Each panel shows the histogram (left) and the estimated log-DOS against the exact
# reference (right) over iterations.

cols = palette(:viridis, max(n_iter, 2))
i0   = searchsortedlast(centers_sum, 0.0)
p1 = plot(xlabel = "Sₙ", ylabel = "counts", title = "histograms", legend = false)
for it in 1:n_iter
    plot!(p1, centers_sum, iter_hist[:, it]; lw = 2, color = cols[it])
end
p2 = plot(xlabel = "Sₙ", ylabel = "-log w", title = "log-DOS vs exact", legend = :topright)
for it in 1:n_iter
    plot!(p2, centers_sum, -iter_lw[:, it] .+ iter_lw[i0, it]; lw = 2, color = cols[it], label = "")
end
plot!(p2, centers_sum, logpdf_sum .- logpdf_sum[i0]; lw = 2, color = :black, ls = :dash, label = "exact")
plot(p1, p2; layout = (1, 2), size = (900, 280), margin = 4Plots.mm)

# ## Reweighting to a biased ensemble
#
# A single multicanonical run can be reweighted post-hoc to any biased ensemble
# ``\beta``, recovering the same result as a dedicated Metropolis run at that ``\beta``.

β_rw    = 0.5
hist_rw = iter_hist[:, end] .* exp.(-β_rw .* centers_sum .- iter_lw[:, end])
dist_rw = hist_rw ./ (sum(hist_rw) * step(bins_sum))

plot(centers_sum, dist_rw; lw = 2, color = :black, label = "MUCA reweighted β = $β_rw",
     xlabel = "Sₙ", ylabel = "density", title = "reweighting vs direct Metropolis",
     size = (600, 250), margin = 5Plots.mm)
plot!(centers_sum, is_dists[3]; lw = 2, color = :steelblue, ls = :dash, label = "Metropolis β = $β_rw")
