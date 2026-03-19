# %%                                                #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../")) #src
Pkg.instantiate()                                   #src

# # Logo prototype via tri-modal potential and Metropolis chain
#
# Compact script that reproduces the notebook's core figure: three Gaussian
# minima and a Metropolis chain overlay.

using Random, LinearAlgebra, Plots
using MonteCarloX

# Approximate Julia-logo dot centers: left (red), right (purple), top (green)
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

basin_idx(x) = argmin([sum((x .- μ).^2) for μ in minima]);

# Metropolis chain using MonteCarloX
rng = Xoshiro(42)
alg = Metropolis(rng, logdensity)

nsteps = 10_000
burnin = 0
Δ = 0.24

x = [0.0, 0.0]
samples = Matrix{Float64}(undef, 2, nsteps)
basins = Vector{Int}(undef, nsteps)

for t in 1:nsteps
    x_new = x .+ Δ .* randn(rng, 2)
    if accept!(alg, x_new, x)
        x .= x_new
    end
    samples[:, t] .= x
    basins[t] = basin_idx(x)
end

keep = (burnin + 1):nsteps
xs = samples[1, keep]
ys = samples[2, keep]
bs = basins[keep]

println("acceptance rate = ", round(acceptance_rate(alg), digits=3))

# ## Plotting
default(
    size=(600, 600),
    legend=false,
    grid=false,
    framestyle=:none,
    background_color=:transparent,
    background_color_outside=:transparent,
    dpi=300,
)

xg = range(-2.0, 2.0; length=420)
yg = range(-1.8, 2.1; length=420)

Ugrid = [-logdensity([x, y]) for y in yg, x in xg]
Bgrid = [basin_idx([x, y]) for y in yg, x in xg]

umin = minimum(Ugrid)
levels_inner = umin .+ [0.3, 1.0]
levels_outer = umin .+ [2.25]
x_min = minimum(xs)
x_max = maximum(xs)
y_min = minimum(ys)
y_max = maximum(ys)

p = plot(xlim=(x_min, x_max), ylim=(y_min, y_max), aspect_ratio=:equal)

# Global contour for merged overall shape
contour!(p, xg, yg, Ugrid; levels=levels_outer, linewidth=1.3, linecolor=:gray72)

# Basin-colored contours near minima
for b in 1:3
    mask = map(v -> v == b, Bgrid)
    Ub = ifelse.(mask, Ugrid, NaN)
    contour!(p, xg, yg, Ub; levels=levels_inner, linewidth=2.2, linecolor=julia_colors[b])
end

# Single Markov chain trajectory, more apparent
plot!(p, xs, ys; lw=1.4, color=:gray28, alpha=0.30)

# Fewer displayed samples
stride = 8
idx_show = 1:stride:length(xs)
x_show = xs[idx_show]
y_show = ys[idx_show]
b_show = bs[idx_show]

# Color only samples in clear proximity of minima; others stay gray
r_clear = 0.52
near = [minimum(sum((([x, y] .- μ).^2)) for μ in minima) <= r_clear^2 for (x, y) in zip(x_show, y_show)]

for b in 1:3
    idx_b = findall(i -> near[i] && b_show[i] == b, eachindex(b_show))
    scatter!(p, x_show[idx_b], y_show[idx_b]; ms=4.6, color=julia_colors[b], alpha=0.57)
end

idx_gray = findall(i -> !near[i], eachindex(near))
scatter!(p, x_show[idx_gray], y_show[idx_gray]; ms=4.0, color=:gray58, alpha=0.36)

plot!(p, axis=nothing, ticks=nothing, border=:none)

p