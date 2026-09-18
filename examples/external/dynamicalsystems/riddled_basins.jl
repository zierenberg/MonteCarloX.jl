# # Riddled basins: confinement and temperature
#
# Some dynamical systems have two attractors whose basins are **riddled** through each other
# [Alexander, Yorke, Kan 1992]: every neighborhood of every point contains pieces of both basins,
# at every scale. This script asks the natural question for a local Monte Carlo search: if you
# confine a random walk to one target attractor — accept a step only if it lands back on the
# target, reject it otherwise — how far can it actually wander before that confinement breaks down?
# And if you relax "reject otherwise" into a temperature that tolerates *some* fraction of
# wrong-attractor visits, how much wrongness does it take to loosen the walk up?
#
# Run: `julia examples/external/dynamicalsystems/riddled_basins.jl`
import Pkg; Pkg.activate(@__DIR__)
using Random, Statistics, Printf
using Attractors, DynamicalSystems, OrdinaryDiffEqVerner
using MonteCarloX
using Plots

# ## The system
#
# The bistable "modified Lorenz" system from the
# [Attractors.jl tutorial](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/attractors/stable/tutorial/):
# a chaotic attractor coexists with a second, smaller attractor. `BasinMapRecurrences` classifies
# any initial condition into an integer attractor `id` by integrating until the trajectory settles
# into a previously located attractor — one full ODE integration per `id(u0)` call.

function modified_lorenz_rule(u, p, _)
    x, y, z = u; a, b = p
    dx = y - x
    dy = -x * z + b * abs(z)
    dz = x * y - a
    return SVector(dx, dy, dz)
end

p0 = [5.0, 0.1]
diffeq = (alg=Vern9(), abstol=1.0e-9, reltol=1.0e-9, dt=0.01)
ds = CoupledODEs(modified_lorenz_rule, [-4.0, 5, 0], p0; diffeq)
grid = (range(-10.0, 10.0; length=150), range(-15.0, 15.0; length=150), range(-15.0, 15.0; length=150))
lo, hi = first.(grid), last.(grid)
clip(x) = clamp.(x, lo, hi)

# `BasinMapRecurrences` is a stateful functor: it caches attractors as it discovers them, so it must
# be built once and reused for every classification below, not reconstructed per call.
basin_map = BasinMapRecurrences(ds, grid; consecutive_recurrences=1000, attractor_locate_steps=1000, consecutive_lost_steps=100)
bmap(x) = basin_map(x; show_progress=false)  # default show_progress=true floods stdout over ~10^4 calls
idcolor = Dict(1 => "#7143E0", 2 => "#191E44")   # Attractors.jl's own palette for these two ids
target = 1
rng = Xoshiro(7)
random_ic(rng) = clip([20 * (rand(rng) - 0.5), 30 * (rand(rng) - 0.5), 30 * (rand(rng) - 0.5)])
nothing #hide

# ## What the boundary looks like
#
# Fix `z` and classify a whole `(x, y)` plane. A smooth boundary would trace a clean dividing
# curve; this shows both colors finely intermixed almost everywhere in this slice.

xs, ys = range(lo[1], hi[1]; length=150), range(lo[2], hi[2]; length=150)
slice = [bmap([x, y, 0.0]) for x in xs, y in ys]
heatmap(xs, ys, permutedims(Float64.(slice)); color=cgrad([idcolor[1], idcolor[2]]), colorbar=false,
        xlabel="x", ylabel="y", title="basin id, fixed z=0 slice", titlefontsize=11,
        size=(550, 480), margin=5Plots.mm)

# ## Rejection sampling, for reference
#
# Draw a fresh random point, keep it if `id(u0) == target`. Cheap, unbiased, and the natural
# reference point for everything below.

n_rej = 2000
rej_hits = count(==(target), [bmap(random_ic(rng)) for _ in 1:n_rej])
@printf("rejection sampling: %d / %d hits (%.0f%% base rate)\n\n", rej_hits, n_rej, 100rej_hits / n_rej)

# ## Confined to the target: how far can it step?
#
# `MetropolisAlgorithm` with `BoltzmannEnsemble`'s "energy" `E(x) = 0` on the target, `1` off it: by
# construction `id ∈ {1, 2}` and `mismatch(x) = 0` exactly when `id(x) == target`, so the target
# attractor is the *unique* minimum-energy state — nothing else can accidentally tie for it.
# `accept!` at temperature `T = 0` then accepts a proposal exactly when it lands back on the target,
# and rejects (stays put) otherwise — precisely "accept if target, reject if not". Sweep the
# proposal step size `σ` and watch the acceptance rate: a smooth boundary would tolerate any `σ`
# smaller than the distance to it; a riddled one starts rejecting almost immediately.
#
# (`T = 0` is a genuine zero-temperature limit, `β = 1/T = ∞`. Two `Inf * 0.0 = NaN` traps had to be
# fixed in MCX core for this to "just work": `BoltzmannEnsemble.logweight` special-cases `E == 0`
# (comparing two on-target states, `E_new = E_old = 0`, otherwise gives `Inf * 0.0`), and the general
# `accept!(alg, arg_new, arg_old)` now checks `linear_logweight(ens)` and, when it holds, forms the
# ratio as `logweight(arg_new - arg_old)` rather than `logweight(arg_new) - logweight(arg_old)` — the
# latter hits `-Inf - (-Inf) = NaN` at `T = 0` whenever *both* energies are nonzero, even though their
# difference is perfectly well-defined. This is a general fix, not special-cased to
# `BoltzmannEnsemble`: any current or future linear ensemble gets it automatically. Both changes are
# exact for every `T`, not just patches for `T = 0`, and cost nothing — `linear_logweight` is a trait
# of the ensemble's *type*, so the branch it picks is resolved and the other one deleted at compile
# time; `@code_llvm` confirms no trace of the check survives, and `@btime` shows both paths
# indistinguishable from their unspecialized originals, 0 allocations either way.)

mismatch(x) = bmap(x) == target ? 0.0 : 1.0
@assert Set(unique(bmap(random_ic(rng)) for _ in 1:500)) ⊆ Set([1, 2]) "unexpected attractor id"

function random_target_ic(rng)
    x = random_ic(rng)
    while bmap(x) != target
        x = random_ic(rng)
    end
    return x
end

function confined_walk(rng, T, nsteps, σ, x0)
    alg = MetropolisAlgorithm(rng, BoltzmannEnsemble(T=T))
    x = copy(x0)
    Ex = mismatch(x)
    n_ontarget, n_accept = 0, 0
    xs = Vector{Vector{Float64}}()   # on-target visits only (see the visual section below)
    for i in 1:nsteps
        xnew = clip(x .+ σ .* randn(rng, 3))
        Enew = mismatch(xnew)
        if accept!(alg, Enew, Ex)
            x, Ex = xnew, Enew
            n_accept += 1
        end
        if Ex == 0.0
            n_ontarget += 1
            push!(xs, copy(x))
        end
    end
    return xs, n_accept / nsteps, n_ontarget / nsteps
end

# Every sweep below starts from the *same* point `x0shared`, and every reported number is averaged
# over a handful of independent repeats from it: different random starting points (and different
# random continuations) sit in differently-riddled neighborhoods — riddling is dense but not
# spatially uniform — which would otherwise swamp the σ- and T-dependence we actually want to see
# with point-to-point noise.
x0shared = random_target_ic(rng)
meanover(f, reps) = mean(f() for _ in 1:reps)

nsteps, reps = 1500, 8
println("strict confinement (T = 0), acceptance rate vs. step size:")
for σ in [0.02, 0.05, 0.1, 0.3, 0.5, 1.0]
    acc_rate = meanover(reps) do
        _, acc, _ = confined_walk(rng, 0.0, nsteps, σ, x0shared)
        acc
    end
    @printf("  σ=%.2f   acceptance rate=%.3f\n", σ, acc_rate)
end

# Under strict confinement `id(x)` never actually leaves the target (that's the whole point of
# rejecting every move that would), so the interesting number is how often a *proposal* survives at
# all. Broadly, larger steps get rejected more — riddling made concrete, not just visible in the
# slice above — but don't expect a smooth curve: at the smallest scales the individual numbers can
# jump around (`σ = 0.05` sometimes lands *higher* than `σ = 0.02`), which is itself a symptom of a
# genuinely fractal boundary rather than noise to average away — self-similar structure has no
# reason to vary smoothly with scale. Even at `σ = 1.0`, a large step relative to the ~20-unit box,
# acceptance never approaches the near-100% a smooth, safely-distant boundary would give.

# ## Temperature: tolerating some fraction of the wrong attractor
#
# Replace the strict rule with a genuinely finite temperature: now a step onto the wrong attractor
# is accepted with probability `exp(-1/T)` instead of almost never. Fix `σ` and sweep `T` up from
# strict confinement to no confinement at all (`T = ∞`, which ignores `id` entirely and reduces to a
# plain local random walk).

σ = 0.1
println("\nfixed step size σ=$σ, on-target fraction vs. temperature:")
for T in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, Inf]
    purity = meanover(reps) do
        _, _, p = confined_walk(rng, T, nsteps, σ, x0shared)
        p
    end
    @printf("  T=%-5s   on-target fraction=%.3f\n", T, purity)
end

# `T = ∞` recovers essentially the same on-target fraction as plain rejection sampling above —
# exactly as it should, since with no confinement at all the walk no longer uses `id` for anything,
# and is just as likely to be on the target as a fresh random draw. Between the two extremes,
# temperature is a direct, continuous knob on "what fraction of wrong-attractor visits am I willing
# to tolerate", smoothly trading purity for how loosely the walk is tied to `id(u0)`.

# ## Visual comparison: how much of the basin does each method actually cover?
#
# Plot the rejection-sampled cloud from earlier (independent draws, so it traces out the target
# basin's true extent) underneath the confined walk's own visited positions, started from the same
# `x0shared` used above so any difference in spread is down to `T` alone. Only on-target positions
# are shown for the walk, matching the rejection cloud's own restriction to `id(u0) == target`.

rej_cloud = Vector{Vector{Float64}}()
while length(rej_cloud) < 600
    x = random_ic(rng)
    bmap(x) == target && push!(rej_cloud, x)
end

plots = map([0.0, 0.3, 1.0, 2.0, 5.0, Inf]) do T
    xs, _, purity = confined_walk(rng, T, 3000, σ, x0shared)
    p = scatter([c[1] for c in rej_cloud], [c[2] for c in rej_cloud]; color=:gray80,
                markersize=2, markerstrokewidth=0, label="rejection cloud")
    scatter!(p, [x[1] for x in xs], [x[2] for x in xs]; color=idcolor[target],
             markersize=2, markerstrokewidth=0, alpha=0.5, label="walk, T=$T",
             xlabel="x", ylabel="y", xlims=(lo[1], hi[1]), ylims=(lo[2], hi[2]), titlefontsize=10,
             title="T=$T  ($(round(Int, 100purity))% on target)")
end
plot(plots...; layout=(2, 3), size=(1500, 800), margin=5Plots.mm,
     plot_title="confined walk vs. rejection cloud, attractor $target only")

# `T = 0` sits at 100% on target by construction (it can never accept a move away); `T = ∞` lands at
# the same on-target percentage as plain rejection sampling, since at infinite temperature the walk
# never uses `id` in its acceptance decision at all and is exactly as likely to be on the target as a
# fresh random draw. Everything in between traces out the purity/mobility tradeoff numerically swept
# above, now visible directly in how much of the gray cloud each colored walk actually covers.

# ## References
#
# - J. C. Alexander, J. A. Yorke, Z. You, I. Kan, *Riddled basins*, Int. J. Bifurcation Chaos
#   **2**, 795 (1992). [doi:10.1142/S0218127492000446](https://doi.org/10.1142/S0218127492000446)
# - C. Grebogi, E. Ott, J. A. Yorke, *Fractal basin boundaries, long-lived chaotic transients, and
#   unstable-unstable pair bifurcation*, Phys. Rev. Lett. **50**, 935 (1983).
#   [doi:10.1103/PhysRevLett.50.935](https://doi.org/10.1103/PhysRevLett.50.935)
# - G. Datseris, A. Wagemakers, *Effortless estimation of basins of attraction*, Chaos **32**, 023104
#   (2022). [doi:10.1063/5.0076568](https://doi.org/10.1063/5.0076568)
