# # Large Deviations of the Ornstein–Uhlenbeck Process
#
# The [Ornstein–Uhlenbeck (OU) process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
# is a continuous-time stochastic process with mean-reverting drift:
#
# ```math
# dx_t = \theta(\mu - x_t)\,dt + \sigma\,dW_t
# ```
#
# where ``\mu`` is the long-time mean, ``\theta`` the inverse relaxation timescale,
# ``\sigma = \sqrt{2D}`` the noise amplitude, and ``dW_t \sim \mathcal{N}(0, dt)`` a
# Wiener increment. The terminal value ``x(T)`` is Gaussian with known mean and
# variance, an exact reference for three sampling strategies: direct sampling,
# biased Metropolis, and multicanonical iteration.

using Random, Distributions, StatsBase, LinearAlgebra, DelimitedFiles
using MonteCarloX
using Plots

datadir     = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "data")))  # hide
traj_file   = joinpath(datadir, "muca_OU_trajectories.tsv")  # hide
ts_file     = joinpath(datadir, "muca_OU_timeseries.tsv")    # hide
dist_file   = joinpath(datadir, "muca_OU_distributions.tsv") # hide
iterh_file  = joinpath(datadir, "muca_OU_iter_hist.tsv")     # hide
iterw_file  = joinpath(datadir, "muca_OU_iter_logweight.tsv")# hide
accept_file = joinpath(datadir, "muca_OU_acceptance.tsv")    # hide

μ, D, θ    = 0.0, 1.0, 1.0
dt, T, x0  = 0.1, 10.0, 0.0
n_direct     = 100_000
n_therm      = 10_000
n_metro      = 1_000_000
n_muca       = 10_000_000
n_iter       = 10
n_iter_steps = 5_000_000
βs           = [0.0, 3.0, 6.0]

## exact terminal distribution (see Wikipedia: OU process)
mean_T = x0 * exp(-T * θ) + μ * (1 - exp(-T * θ))
var_T  = θ < 1e-6 ? 2 * D * T : D / θ * (1 - exp(-2 * T * θ))
std_T  = sqrt(var_T)
pdf_T  = Normal(mean_T, std_T)
bins_T    = (mean_T - 10 * std_T):(std_T / 10):(mean_T + 10 * std_T)
centers_T = collect(bins_T[1:end-1] .+ diff(collect(bins_T)) ./ 2)
logpdf_T  = logpdf.(pdf_T, centers_T)
density(w) = w ./ (sum(w) * step(bins_T))
nothing # hide

# ## Trajectory system
#
# The state is a full discretised trajectory ``\{dW_t\}`` of Wiener increments.
# Each update proposes a change to one increment, accepts it under the Gaussian
# prior (Metropolis-within-Gibbs), then accepts or rejects on the terminal value
# ``x(T)`` via the sampling algorithm.

mutable struct OUTrajectory
    ts        :: Vector{Float64}
    xs        :: Vector{Float64}
    dWs       :: Vector{Float64}
    tmp_xs    :: Vector{Float64}
    tmp_dWs   :: Vector{Float64}
    logpdf_dW :: Function
    μ :: Float64;  σ :: Float64
    θ :: Float64;  dt :: Float64
    function OUTrajectory(rng; x0 = 0.0, μ = 0.0, D = 1.0, θ = 1.0, dt = 0.1, T = 10.0)
        N    = round(Int, T / dt) + 1
        σ    = sqrt(2 * D)
        dist = Normal(0, sqrt(dt))
        dWs  = rand(rng, dist, N)
        xs   = zeros(N)
        ts   = collect(0:dt:(N-1)*dt)
        sys  = new(ts, xs, dWs, zeros(N), zeros(N), x -> logpdf(dist, x), μ, σ, θ, dt)
        sys.xs[1] = x0
        integrate!(sys)
        return sys
    end
end

@inline function integrate!(sys::OUTrajectory, r::UnitRange{Int} = 1:0)
    r = isempty(r) ? (1:length(sys.xs)) : r
    x = sys.xs[first(r)]
    for i in first(r):last(r)-1
        x += sys.θ * (sys.μ - x) * sys.dt + sys.σ * sys.dWs[i]
        sys.xs[i+1] = x
    end
end

x_T(sys::OUTrajectory) = sys.xs[end]

function update!(sys::OUTrajectory, alg::AbstractMarkovChainMonteCarlo; δ = 0.5)
    idx    = rand(alg.rng, 1:length(sys.dWs))
    dW_old = sys.dWs[idx]
    dW_new = dW_old + δ * (2 * rand(alg.rng) - 1)
    if rand(alg.rng) < exp(sys.logpdf_dW(dW_new) - sys.logpdf_dW(dW_old))   # accept under prior first
        sys.tmp_xs[idx:end]  .= sys.xs[idx:end]
        sys.tmp_dWs[idx:end] .= sys.dWs[idx:end]
        sys.dWs[idx] = dW_new
        integrate!(sys, idx:length(sys.xs))
        if !accept!(alg, sys.xs[end], sys.tmp_xs[end])
            sys.xs[idx:end]  .= sys.tmp_xs[idx:end]
            sys.dWs[idx:end] .= sys.tmp_dWs[idx:end]
        end
    else
        alg.steps += 1
    end
end

# ## Running the samplers
#
# Every strategy below is expensive (up to ``5\times10^7`` steps), so we run them
# once and cache the trajectories, time series and distributions; a docs build only
# reloads. **Direct sampling** draws independent terminal values. **Biased Metropolis**
# tilts by ``e^{\beta x(T)}`` to reach the tails. **Multicanonical** sampling learns
# flat weights so the whole support is visited uniformly.

if !isfile(dist_file)                                          # hide
## single reference trajectory
sys0 = OUTrajectory(MersenneTwister(1234); x0 = x0, μ = μ, D = D, θ = θ, dt = dt, T = T)

## direct sampling of x(T)
direct_samples = [OUTrajectory(MersenneTwister(i); x0 = x0, μ = μ, D = D, θ = θ,
                               dt = dt, T = T).xs[end] for i in 1:n_direct]
dist_direct = normalize(fit(Histogram, direct_samples, bins_T); mode = :pdf).weights

## biased Metropolis at several β — keep the x(T) time series, histogram, final trajectory
function run_metropolis(β)
    sys  = OUTrajectory(MersenneTwister(123); x0 = x0, μ = μ, D = D, θ = θ, dt = dt, T = T)
    alg  = Metropolis(MersenneTwister(42); β = β)
    meas_ts   = Measurements([:x_T => x_T => Float64[]], interval = 100)
    meas_hist = Measurements([:x_T => x_T => fit(Histogram, [], bins_T)], interval = 1)
    for _ in 1:n_therm; update!(sys, alg); end
    for i in 1:n_metro
        update!(sys, alg)
        measure!(meas_hist, sys, i)
        measure!(meas_ts, sys, i)
    end
    return meas_ts[:x_T].data, density(meas_hist[:x_T].data.weights), sys
end
metro = [run_metropolis(β) for β in βs]
ts_series = hcat((m[1] for m in metro)...)          # x(T) time series per β
is_dists  = [m[2] for m in metro]                    # biased terminal densities

## multicanonical with flat weights
sys_muca0 = OUTrajectory(MersenneTwister(123); x0 = x0, μ = μ, D = D, θ = θ, dt = dt, T = T)
alg_muca0 = Multicanonical(MersenneTwister(100), BinnedObject(bins_T, 0.0))
meas_muca0 = Measurements([:x_T => x_T => fit(Histogram, [], bins_T)], interval = 1)
for _ in 1:n_therm; update!(sys_muca0, alg_muca0); end
for i in 1:n_muca;  update!(sys_muca0, alg_muca0); measure!(meas_muca0, sys_muca0, i); end
dist_flat = density(meas_muca0[:x_T].data.weights)

## multicanonical iteration
sys_iter = OUTrajectory(MersenneTwister(42); x0 = x0, μ = μ, D = D, θ = θ, dt = dt, T = T)
alg_iter = Multicanonical(MersenneTwister(42), BinnedObject(bins_T, 0.0))
x_left  = first(bins_T) + std_T
x_right = last(bins_T)  - std_T
cs      = get_centers(logweight(alg_iter))
iter_hist   = zeros(length(centers_T), n_iter)
iter_lw     = zeros(length(centers_T), n_iter)
iter_accept = zeros(n_iter)
for it in 1:n_iter
    for _ in 1:n_therm;      update!(sys_iter, alg_iter; δ = 0.5); end
    reset!(alg_iter)
    for _ in 1:n_iter_steps; update!(sys_iter, alg_iter; δ = 0.5); end
    update_logweight!(ensemble(alg_iter); mode = :simple)
    set_logweight!(ensemble(alg_iter), (first(cs), x_left),  x -> logweight(alg_iter)(x_left)  + (x - x_left)  * 2.0)
    set_logweight!(ensemble(alg_iter), (x_right, last(cs)),  x -> logweight(alg_iter)(x_right) - (x - x_right) * 2.0)
    iter_hist[:, it]  = ensemble(alg_iter).histogram.values
    iter_lw[:, it]    = logweight(alg_iter).values
    iter_accept[it]   = acceptance_rate(alg_iter)
end

theader = permutedims(["t"; "x_single"; "dW_single";                       # hide
                       reduce(vcat, [["x_b$(β)", "dW_b$(β)"] for β in βs])])# hide
traj_cols = hcat(sys0.ts, sys0.xs, sys0.dWs,                                # hide
                 reduce(hcat, [hcat(m[3].xs, m[3].dWs) for m in metro]))    # hide
mkpath(datadir)                                                            # hide
writedlm(traj_file, [theader; traj_cols], '\t')                            # hide
writedlm(ts_file, [permutedims(["step"; ["xT_b$(β)" for β in βs]]);         # hide
                   hcat(1:size(ts_series, 1), ts_series)], '\t')           # hide
dheader = permutedims(["x_T"; "direct"; ["is_b$(β)" for β in βs]; "muca_flat"])  # hide
writedlm(dist_file, [dheader;                                              # hide
    hcat(centers_T, dist_direct, is_dists[1], is_dists[2], is_dists[3], dist_flat)], '\t')  # hide
iheader = permutedims(["x_T"; ["iter$(it)" for it in 1:n_iter]])            # hide
writedlm(iterh_file, [iheader; hcat(centers_T, iter_hist)], '\t')          # hide
writedlm(iterw_file, [iheader; hcat(centers_T, iter_lw)], '\t')            # hide
writedlm(accept_file, ["iter" "acceptance"; hcat(1:n_iter, iter_accept)], '\t')  # hide
end                                                                         # hide
tr = readdlm(traj_file, '\t'; header = true)[1]                            # hide
tsd = readdlm(ts_file, '\t'; header = true)[1]                             # hide
dd = readdlm(dist_file, '\t'; header = true)[1]                            # hide
ih = readdlm(iterh_file, '\t'; header = true)[1]                           # hide
iw = readdlm(iterw_file, '\t'; header = true)[1]                           # hide
ac = readdlm(accept_file, '\t'; header = true)[1]                          # hide
traj_t = tr[:, 1]; traj = tr[:, 2:end]                                     # hide
ts_series = tsd[:, 2:end]                                                  # hide
dist_direct = dd[:, 2]; is_dists = [dd[:, 3], dd[:, 4], dd[:, 5]]; dist_flat = dd[:, 6]  # hide
iter_hist = ih[:, 2:end]; iter_lw = iw[:, 2:end]; iter_accept = ac[:, 2]   # hide
n_iter = size(iter_hist, 2)                                                # hide
nothing # hide

# ## A single trajectory
#
# One realisation of the OU process relaxing toward its mean.

plot(traj_t, traj[:, 1]; label = nothing, xlabel = "time", ylabel = "x",
     title = "single OU trajectory", size = (700, 220), margin = 5Plots.mm)

# ## Direct sampling
#
# Independent trajectories reproduce the bulk of the terminal distribution.

plot(centers_T, dist_direct; st = :bar, linewidth = 0, alpha = 0.6, label = "direct",
     xlabel = "x(T)", ylabel = "density", title = "terminal distribution vs exact",
     size = (600, 250), margin = 5Plots.mm)
plot!(centers_T, pdf.(pdf_T, centers_T); lw = 2, color = :black, ls = :dash, label = "exact")

# ## Biased Metropolis
#
# A biasing field ``\beta`` enhances trajectories ending at large ``x(T)``, giving
# direct access to the tails. The time series shows how each ``\beta`` shifts the
# explored region.

p_ts = plot(xlabel = "iteration", ylabel = "x(T)", title = "time series",
            legend = :topleft, size = (600, 220), margin = 5Plots.mm)
for (k, β) in enumerate(βs)
    plot!(p_ts, ts_series[:, k]; label = "β = $β", lw = 1)
end
p_ts

# Together the biased runs cover the full distribution, including the exponentially
# rare large-``x(T)`` events.

p_dist = plot(centers_T, pdf.(pdf_T, centers_T); lw = 2, color = :black, ls = :dash,
              label = "exact", xlabel = "x(T)", ylabel = "density",
              title = "biased distributions", size = (600, 250), margin = 5Plots.mm)
for (k, β) in enumerate(βs)
    plot!(p_dist, centers_T, is_dists[k]; lw = 2, label = "β = $β")
end
p_dist

# The corresponding final trajectories and Wiener increments: larger ``\beta``
# systematically shifts the ensemble toward higher terminal values.

p_traj = plot(xlabel = "time", ylabel = "x", title = "typical trajectories per β",
              size = (700, 260), margin = 5Plots.mm)
p_dW   = plot(xlabel = "time", ylabel = "dW", title = "Wiener increments",
              size = (700, 260), margin = 5Plots.mm)
for (k, β) in enumerate(βs)
    plot!(p_traj, traj_t, traj[:, 2 + 2 * (k - 1) + 1]; label = "β = $β")
    plot!(p_dW,   traj_t, traj[:, 2 + 2 * (k - 1) + 2]; label = "β = $β")
end
plot(p_traj, p_dW; layout = (2, 1), size = (700, 420), margin = 5Plots.mm)

# ## Multicanonical — flat weights
#
# With flat weights the algorithm samples ``x(T)`` uniformly across the full support.

plot(centers_T, dist_flat; lw = 2, label = "MUCA (flat)",
     xlabel = "x(T)", ylabel = "density", title = "flat-weight multicanonical",
     size = (600, 250), margin = 5Plots.mm)
plot!(centers_T, pdf.(pdf_T, centers_T); lw = 2, color = :black, ls = :dash, label = "exact")

# ## Multicanonical iteration
#
# Starting from flat weights, the algorithm iteratively learns the log-DOS; linear
# boundary tails keep the chain inside the binned region. The acceptance rate
# stabilises and the estimate converges onto the exact reference.

plot(iter_accept; xlabel = "iteration", ylabel = "acceptance rate",
     title = "convergence of acceptance rate", label = nothing,
     size = (600, 220), margin = 5Plots.mm)

cols = palette(:viridis, max(n_iter, 2))
i0   = searchsortedlast(centers_T, 0.0)
p1 = plot(xlabel = "x(T)", ylabel = "counts", title = "histograms", legend = false)
for it in 1:n_iter
    plot!(p1, centers_T, iter_hist[:, it]; lw = 2, color = cols[it])
end
p2 = plot(xlabel = "x(T)", ylabel = "-log w", title = "log-DOS vs exact", legend = :topright)
for it in 1:n_iter
    plot!(p2, centers_T, -iter_lw[:, it] .+ iter_lw[i0, it]; lw = 2, color = cols[it], label = "")
end
plot!(p2, centers_T, logpdf_T .- logpdf_T[i0]; lw = 2, color = :black, ls = :dash, label = "exact")
plot(p1, p2; layout = (1, 2), size = (900, 280), margin = 4Plots.mm)
