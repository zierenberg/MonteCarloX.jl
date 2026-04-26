# %%                                                #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../"))         #src
Pkg.instantiate()                                   #src

# # Wide banner: "MonteCarloX.jl" + tri-modal logo
#
# Generates a wide banner PNG for the README. The logo graphic (right)
# is the same tri-modal Metropolis chain as logo.jl.

using Random, LinearAlgebra, Plots
using MonteCarloX

# ── Tri-modal potential ──────────────────────────────────────────────

h = 1.0
base = 2*h/sqrt(3)
const minima = [
    [-base/2, 0.0],
    [ base/2, 0.0],
    [    0.0,   h],
]

const julia_colors = [:"#CB3C33", :"#9558B2", :"#389826"]

const σ = 0.30
const κ = 0.02

logsumexp(v) = begin
    m = maximum(v)
    m + log(sum(exp.(v .- m)))
end

component_logweight(x, μ; σ=σ) = -0.5 * sum((x .- μ).^2) / σ^2

function logdensity(x)
    comps = [component_logweight(x, μ) for μ in minima]
    return logsumexp(comps) - κ * sum(x .^ 2)
end

basin_idx(x) = argmin([sum((x .- μ).^2) for μ in minima])

# ── Metropolis chain ─────────────────────────────────────────────────

rng = Xoshiro(42)
alg = Metropolis(rng, logdensity)

nsteps = 10_000
Δ = 0.24

x = [0.0, 0.0]
samples = Matrix{Float64}(undef, 2, nsteps)
basins  = Vector{Int}(undef, nsteps)

for t in 1:nsteps
    x_new = x .+ Δ .* randn(rng, 2)
    if accept!(alg, x_new, x)
        x .= x_new
    end
    samples[:, t] .= x
    basins[t] = basin_idx(x)
end

xs = samples[1, :]
ys = samples[2, :]
bs = basins

# ── Banner layout ────────────────────────────────────────────────────

default(
    legend=false,
    grid=false,
    framestyle=:none,
    background_color=:transparent,
    background_color_outside=:transparent,
    dpi=300,
)

layout = @layout [a{0.70w} b{0.30w}]
banner = plot(layout=layout, size=(1600, 450), margin=0Plots.mm)

# ── Left panel: title text ───────────────────────────────────────────

plot!(banner[1],
    xlim=(0, 1), ylim=(0, 1),
    framestyle=:none, axis=nothing, ticks=nothing,
    margin=0Plots.mm,
)

annotate!(banner[1], 0.5, 0.5,
    text("MonteCarloX.jl", 100, :gray50, :center))

# ── Right panel: logo graphic ────────────────────────────────────────

xg = range(-2.0, 2.0; length=420)
yg = range(-1.8, 2.1; length=420)

Ugrid = [-logdensity([xi, yi]) for yi in yg, xi in xg]
Bgrid = [basin_idx([xi, yi]) for yi in yg, xi in xg]

umin = minimum(Ugrid)
levels_inner = umin .+ [0.3, 1.0]
levels_outer = umin .+ [2.25]

plot!(banner[2],
    xlim=(minimum(xs), maximum(xs)),
    ylim=(minimum(ys), maximum(ys)),
    aspect_ratio=:equal,
    framestyle=:none, axis=nothing, ticks=nothing,
    margin=0Plots.mm,
)

contour!(banner[2], xg, yg, Ugrid;
    levels=levels_outer, linewidth=1.3, linecolor=:gray72)

for b in 1:3
    mask = map(v -> v == b, Bgrid)
    Ub = ifelse.(mask, Ugrid, NaN)
    contour!(banner[2], xg, yg, Ub;
        levels=levels_inner, linewidth=2.2, linecolor=julia_colors[b])
end

plot!(banner[2], xs, ys; lw=1.4, color=:gray28, alpha=0.30)

stride = 8
idx_show = 1:stride:length(xs)
x_show = xs[idx_show]
y_show = ys[idx_show]
b_show = bs[idx_show]

r_clear = 0.52
near = [minimum(sum((([xi, yi] .- μ).^2)) for μ in minima) <= r_clear^2
        for (xi, yi) in zip(x_show, y_show)]

for b in 1:3
    idx_b = findall(i -> near[i] && b_show[i] == b, eachindex(b_show))
    scatter!(banner[2], x_show[idx_b], y_show[idx_b];
        ms=4.6, color=julia_colors[b], alpha=0.57)
end

idx_gray = findall(i -> !near[i], eachindex(near))
scatter!(banner[2], x_show[idx_gray], y_show[idx_gray];
    ms=4.0, color=:gray58, alpha=0.36)

banner

savefig(banner, joinpath(@__DIR__, "banner.png"))
println("Saved banner.png")
