# # Sampling rare near-regular orbits of a chaotic map with Wang-Landau
#
# A 4-dimensional coupled standard map looks chaotic almost everywhere but hides a pair of
# genuinely rare regular islands (invariant tori). This reproduces the system and parameters from
# Kitajima & Iba, *Multicanonical Sampling of Rare Trajectories in Chaotic Dynamical Systems*
# [arXiv:1003.2013](https://arxiv.org/abs/1003.2013), using MCX's `WangLandauAlgorithm`.
#
# Run: `julia examples/external/dynamicalsystems/forgetting_time.jl`
import Pkg; Pkg.activate(@__DIR__)
using Random, LinearAlgebra, Printf
using DynamicalSystems
using MonteCarloX
using Plots

# ## The map
#
# Two 2D standard maps, `(u,v)` and `(x,y)`, coupled with strength `b`:
# ```
# u' = u - (K/2π)sin(2πv) + (b/2π)sin(2π(v+y)),   v' = v + u'
# x' = x - (K/2π)sin(2πy) + (b/2π)sin(2π(v+y)),   y' = y + x'
# ```
# All four coordinates on `[0,1)`. `b` sets the island size, independent of `K`. We use
# `(K,b) = (6.0, 0.1)`, one of the paper's own choices.

function map4d_rule(z, p, _)
    K, b = p
    u, v, x, y = z
    s = 2π
    up = u - (K/s)*sin(s*v) + (b/s)*sin(s*(v+y))
    vp = v + up
    xp = x - (K/s)*sin(s*y) + (b/s)*sin(s*(v+y))
    yp = y + xp
    return SVector(mod(up, 1.0), mod(vp, 1.0), mod(xp, 1.0), mod(yp, 1.0))
end

# Jacobian, derived analytically and verified against automatic differentiation and against
# `det(J) = 1` (volume preservation) at 2000 random points:

function map4d_jacobian(z, p, _)
    K, b = p
    u, v, x, y = z
    s = 2π
    A = -K*cos(s*v) + b*cos(s*(v+y))
    B = b*cos(s*(v+y))
    C = -K*cos(s*y) + b*cos(s*(v+y))
    return SMatrix{4,4}(1.0,1.0,0.0,0.0,  A,1+A,B,B,  0.0,0.0,1.0,1.0,  B,B,C,1+C)
end

const K, b = 6.0, 0.1
const ds    = DeterministicIteratedMap(map4d_rule, [0.1, 0.2, 0.3, 0.4], (K, b))
const tands = TangentDynamicalSystem(ds; J=map4d_jacobian, k=1)   # 1 tangent (deviation) vector
nothing #hide

# ## The observable: forgetting time
#
# Evolve a random deviation vector under the map's linearization; `T(u0)` is the first iteration at
# which its accumulated log-growth exceeds `-log(ε)`. Large `T` means low chaoticity — near a
# regular island. `ε = 2⁻⁴³` is the paper's threshold; `Tmax` caps the search.

function forgetting_time(rng, u0; ε=2.0^-43, Tmax=250)
    v0 = randn(rng, 4); v0 ./= norm(v0)
    reinit!(tands, u0; Q0=reshape(v0, 4, 1))
    loggrowth, thresh = 0.0, -log(ε)
    for t in 1:Tmax
        DynamicalSystems.step!(tands)
        dv = current_deviations(tands)
        n  = norm(dv)
        loggrowth += log(n)
        set_deviations!(tands, dv ./ n)          # renormalize to avoid overflow
        loggrowth > thresh && return t
    end
    return Tmax
end
const Tmax = 250
nothing #hide

# ## Goal
#
# We want `P(T)`. Wang-Landau learns a bias, from one random walk over `u0`, that makes every `T`
# equally likely to be proposed — that bias **is** `P(T)`, up to normalization.

# ## A Markov chain over initial conditions
#
# State: `u0 ∈ [0,1)^4` and its current `T`. Propose a small step (wrapped to `[0,1)`), recompute
# `T`, `accept!` decides. `T` has no cheap incremental update, so both absolute values are passed.

mutable struct MapWalker
    u0 :: Vector{Float64}
    T  :: Int
end
MapWalker(rng) = (u0 = rand(rng, 4); MapWalker(u0, forgetting_time(rng, SVector{4}(u0); Tmax)))

function propose!(w::MapWalker, alg; δ=0.03)
    u0_new = mod.(w.u0 .+ δ .* (2 .* rand(alg.rng, 4) .- 1), 1.0)
    T_new  = forgetting_time(alg.rng, SVector{4}(u0_new); Tmax)
    if accept!(alg, T_new, w.T)
        w.u0, w.T = u0_new, T_new
    end
end
sweep!(w, alg, n) = (for _ in 1:n; propose!(w, alg); end)
isflat(alg) = flatness(ensemble(alg).histogram, 0, Tmax; criterion=:mean_over_min) <= 2.0
nothing #hide

# ## Training
#
# Every visit to a `T` value decrements its weight, discouraging revisits. Once the histogram is
# flat, `logf` shrinks and the histogram resets. At convergence `logweight(T) = -log(g(T)) + const`,
# so `P(T) ∝ exp(-logweight(T))` with no further sampling needed.
#
# `isflat` uses a loose criterion (`<= 2.0`, not the usual `1.25`) so it can actually be satisfied:
# shrinking `logf` on a histogram that was never flat biases the weights systematically, not just
# noisily — the sanity check below would catch that.

rng = Xoshiro(42)
w   = MapWalker(rng)
alg = WangLandauAlgorithm(rng, 0:1:Tmax; logf=1.0)

n_iters, nsweeps_per_check, max_checks = 12, 10_000, 100
training_evals = 0
for it in 1:n_iters
    for _ in 1:max_checks
        sweep!(w, alg, nsweeps_per_check)
        global training_evals += nsweeps_per_check
        isflat(alg) && break
    end
    update_logweight!(ensemble(alg))
    it < n_iters && reset!(alg)
end

covered = ensemble(alg).visited
@printf("training done: %d iterations, %d forgetting_time evaluations, %d/%d forgetting-time values ever visited (range %s)\n",
        n_iters, training_evals, count(covered), Tmax + 1, extrema(get_centers(ensemble(alg).logweight)[covered]))

# ## The reconstructed distribution

Ts   = get_centers(ensemble(alg).logweight)[covered]
logP = -ensemble(alg).logweight.values[covered]
logP .-= maximum(logP)
Pnorm = exp.(logP) ./ sum(exp.(logP))

plot(Ts, logP; lw=2, marker=:circle, ms=2, label=nothing,
     xlabel="forgetting time T", ylabel="log P(T)  (shifted)",
     title="4D coupled standard map, K=$K, b=$b — reconstructed distribution",
     size=(650, 320), margin=5Plots.mm)

# Sharp drop by `T ≈ 20`, then a rare tail to `Tmax` — almost 4 orders of magnitude, from one
# training run.

# ## Sanity check: does the reconstruction agree with brute force?
#
# Necessary, not sufficient: compare `P(T ≥ threshold)` against uniform sampling. Evaluations are
# cheap (microseconds), so a many-million-draw reference is still fast.

n_check = 10_000_000
harvest_thresh = 100   # also used below to harvest the WL comparison at the same threshold
naive_Ts = Vector{Int}(undef, n_check)
naive_hits = Vector{Float64}[]   # [u,v,x,y] positions of naive draws with T >= harvest_thresh
for i in 1:n_check
    u0 = rand(rng, 4)
    T  = forgetting_time(rng, SVector{4}(u0); Tmax)
    naive_Ts[i] = T
    T >= harvest_thresh && push!(naive_hits, u0)
end
println("\nsanity check, P(T ≥ threshold): brute force vs. the Wang-Landau reconstruction")
for thr in (50, 200)
    p_naive = count(>=(thr), naive_Ts) / n_check
    p_wl    = sum(Pnorm[Ts .>= thr])
    @printf("  threshold=%-4d  uniform ≈ %.3g   Wang-Landau ≈ %.3g\n", thr, p_naive, p_wl)
end

# ## Harvesting: how much more often does the biased walk find the rare region?
#
# Freeze the trained weights (no longer updated) and keep proposing. The real question: for the
# same number of evaluations, how many rare-region examples does each method produce?
#
# The two islands aren't connected by any `T`-gradient through the chaotic bulk between them, so a
# walk making only small steps can spend its entire budget in whichever one it finds first. A small
# fraction of larger steps fixes this (same acceptance rule, no change to the trained weights) — see
# both islands covered in the figure below, rather than just the one a small-step-only walk finds.

propose_mixed!(w, alg) = propose!(w, alg; δ = rand(alg.rng) < 0.15 ? 0.15 : 0.03)

prod_alg = MetropolisAlgorithm(rng, MulticanonicalEnsemble(ensemble(alg).logweight))
n_production = 1_000_000
thresholds = (50, 100, 200)
wl_hits = zeros(Int, length(thresholds))
harvest = Vector{Float64}[]   # [u,v,x,y] for the most forgetful (large-T) visits, for the figure below
for i in 1:n_production
    propose_mixed!(w, prod_alg)
    for (k, thr) in enumerate(thresholds)
        w.T >= thr && (wl_hits[k] += 1)
    end
    w.T >= harvest_thresh && i % 20 == 0 && push!(harvest, copy(w.u0))
end

println("\ndiscovery rate: fraction of proposals/draws with T ≥ threshold")
for (k, thr) in enumerate(thresholds)
    naive_rate = count(>=(thr), naive_Ts) / n_check
    wl_rate    = wl_hits[k] / n_production
    @printf("  T>=%-4d  naive: %-9.3g (%d/%d)   WL: %-9.3g (%d/%d)   -> %.0fx more often\n",
            thr, naive_rate, count(>=(thr), naive_Ts), n_check, wl_rate, wl_hits[k], n_production,
            wl_rate / naive_rate)
end

# Not a better probability estimate (naive already gets that right) — a walk that lands in the rare
# region thousands of times more often per evaluation.
#
# ## Does that hold once training is paid for?
#
# Production alone cost `n_production` evaluations, but training took `training_evals` more to
# build the bias. Total pipeline cost vs. naive reaching the same hit count:

total_wl_evals = training_evals + n_production
println("\nincluding the one-time cost of training ($training_evals evaluations):")
for (k, thr) in enumerate(thresholds)
    naive_rate = count(>=(thr), naive_Ts) / n_check
    naive_evals_needed = wl_hits[k] / naive_rate
    @printf("  T>=%-4d  %d hits cost WL %d evals (train+production); naive would need ~%.3g evals for the same %d hits (%.0fx more)\n",
            thr, wl_hits[k], total_wl_evals, naive_evals_needed, wl_hits[k], naive_evals_needed / total_wl_evals)
end

# Training costs about as much as the naive reference sample itself — but the target is rare enough
# that it still pays for itself many times over. Not true for the single-map version of this
# example: training cost alone doesn't make a method worth it, how rare the target is does.

# ## Where the near-regular initial conditions live
#
# Both sub-map projections, overlaying what each method found at the same threshold.
#
# (Won't match Kitajima & Iba's own figure: they used `b=0.001` for `~10⁻¹²`–`10⁻¹⁴` tori needing
# inset zooms. We used their own `(K,b)=(6.0,0.1)` — rare enough to make the point, fast enough to
# run in a couple of minutes.)

@printf("\n%d WL points and %d naive points plotted below (both T ≥ %d)\n",
        length(harvest), length(naive_hits), harvest_thresh)

p1 = scatter([h[1] for h in harvest], [h[2] for h in harvest]; markersize=2, markerstrokewidth=0,
             alpha=0.4, label="WL harvest", xlabel="u", ylabel="v", xlims=(0,1), ylims=(0,1),
             title="(u,v) projection")
scatter!(p1, [h[1] for h in naive_hits], [h[2] for h in naive_hits]; markersize=3,
         markerstrokewidth=0, alpha=0.6, color=:red, label="naive")
p2 = scatter([h[3] for h in harvest], [h[4] for h in harvest]; markersize=2, markerstrokewidth=0,
             alpha=0.4, label="WL harvest", xlabel="x", ylabel="y", xlims=(0,1), ylims=(0,1),
             title="(x,y) projection")
scatter!(p2, [h[3] for h in naive_hits], [h[4] for h in naive_hits]; markersize=3,
         markerstrokewidth=0, alpha=0.6, color=:red, label="naive")
plot(p1, p2; layout=(1,2), size=(1000, 480), margin=5Plots.mm,
     plot_title="near-regular initial conditions: WL harvest vs. naive (T ≥ $harvest_thresh)")

# ## References
#
# - A. Kitajima, Y. Iba, *Multicanonical Sampling of Rare Trajectories in Chaotic Dynamical
#   Systems*, [arXiv:1003.2013](https://arxiv.org/abs/1003.2013).
# - B. V. Chirikov, *A universal instability of many-dimensional oscillator systems*, Phys.
#   Rep. **52**, 263 (1979). [doi:10.1016/0370-1573(79)90023-1](https://doi.org/10.1016/0370-1573(79)90023-1)
