# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # SIR with Heterogeneous Infectiousness (Lloyd-Smith et al., Nature 2005)
#
# Reproduction of Fig. 2 from Lloyd-Smith et al. (2005).
#
# Individual reproductive number ``\nu`` is Gamma-distributed with mean ``R_0``
# and dispersion parameter ``k``. The number of secondary cases per individual is
#
# ```math
# Z \sim \mathrm{NegBin}(R_0,\, k)
# ```
#
# which is the Poisson mixture over a Gamma-distributed rate (see paper §Methods).
# Smaller ``k`` means more overdispersion (more superspreading).
#
# **Panel 2b** — extinction probability ``q`` is the fixed point of the pgf:
# ```math
# g(s) = \left(1 + \frac{R_0}{k}(1-s)\right)^{-k}, \qquad q = g(q)
# ```
# solved analytically (no simulation required).
#
# **Panel 2c** — outbreak size conditional on non-extinction, simulated as a
# generation-by-generation branching process.
#
# Reference: Lloyd-Smith, J. O. et al. Nature 438, 355–359 (2005).

using Random, Distributions, Plots, Statistics
using MonteCarloX   # branching process uses Xoshiro from here for consistency

const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

# ## Parameters

R0_vals = CI_MODE ? [0.5, 1.5, 3.0] : collect(0.1:0.05:5.0)   # for panel 2b sweep
R0_fig  = 1.5                                                    # fixed R0 for panels 2a, 2c
k_vals  = [0.1, 0.2, 0.5, 1.0, 2.0, Inf]
colors  = [:crimson, :orange, :gold, :steelblue, :mediumpurple, :black]
n_traj  = CI_MODE ? 1_000 : 10_000
threshold = 100   # "outbreak" threshold for panel 2c

# ## Panel 2b: analytical extinction probability
#
# For R0 > 1, q is the unique solution in (0,1) of q = g(q).
# For R0 ≤ 1, q = 1 (certain extinction).
# Solved by simple fixed-point iteration.

function extinction_prob(R0, k; tol=1e-10, maxiter=1000)
    R0 <= 1.0 && return 1.0
    k == Inf && return exp(-(R0 - 1))   # Poisson limit: q = exp(-R0*(1-q)) → q=exp(-(R0-1))

    ## Fixed-point iteration: q_{n+1} = g(q_n)
    ## g(s) = (1 + R0/k*(1-s))^{-k}
    q = 0.5
    for _ in 1:maxiter
        q_new = (1 + R0/k * (1 - q))^(-k)
        abs(q_new - q) < tol && return q_new
        q = q_new
    end
    return q
end

# ## Panel 2c: branching process simulation
#
# Run generation-by-generation. Each infected individual draws
# Z ~ NegBin(R0, k) offspring. Record the first generation at which
# cumulative cases exceed `threshold`, conditional on non-extinction.

function offspring_dist(R0, k)
    k == Inf && return Poisson(R0)
    ## NegativeBinomial(r, p) in Julia: mean = r(1-p)/p, so p = k/(k+R0)
    return NegativeBinomial(k, k / (k + R0))
end

function simulate_branching(rng, dist, threshold; max_gen=500)
    cases = [1]   # generation 0: one index case
    total = 1
    for gen in 1:max_gen
        new_cases = sum(rand(rng, dist) for _ in 1:cases[end])
        push!(cases, new_cases)
        total += new_cases
        new_cases == 0 && return nothing, cases   # extinction
        total >= threshold && return gen, cases   # outbreak
    end
    return nothing, cases   # treat as non-outbreak if threshold not reached
end

# ## Run simulations for panel 2c

rng = Xoshiro(42)

results_c = Dict{Float64, Vector{Int}}()
for k in k_vals
    dist = offspring_dist(R0_fig, k)
    gen_hits = Int[]
    for _ in 1:n_traj
        gen, _ = simulate_branching(rng, dist, threshold)
        gen !== nothing && push!(gen_hits, gen)
    end
    results_c[k] = gen_hits
end

# ## Figures

# ### Panel 2a: Gamma distributions of ν

x = range(0, 6, length=300)
p2a = plot(xlabel="Individual reproductive number ν", ylabel="Probability density",
           title="(a) Distribution of ν  (R₀=$(R0_fig))",
           legend=:topright, size=(500, 300), margin=4Plots.mm)
for (k, col) in zip(k_vals, colors)
    label = k == Inf ? "k = ∞" : "k = $k"
    if k == Inf
        ## Delta function: vertical line at R0
        vline!(p2a, [R0_fig]; color=col, lw=2, label=label)
    else
        d = Gamma(k, R0_fig / k)
        plot!(p2a, x, pdf.(d, x); color=col, lw=2, label=label)
    end
end

# ### Panel 2b: Extinction probability vs R0

p2b = plot(xlabel="R₀", ylabel="Extinction probability q",
           title="(b) Stochastic extinction",
           legend=:topright, ylims=(0,1), size=(500, 300), margin=4Plots.mm)
for (k, col) in zip(k_vals, colors)
    label = k == Inf ? "k = ∞" : "k = $k"
    q_vec = [extinction_prob(R0, k) for R0 in R0_vals]
    plot!(p2b, R0_vals, q_vec; color=col, lw=2, label=label)
end
vline!(p2b, [1.0]; color=:gray, ls=:dash, lw=1, label=nothing)

# ### Panel 2c: Box plots of generation to threshold, conditional on non-extinction

labels_c = [k == Inf ? "k=∞" : "k=$k" for k in k_vals]
p2c = plot(xlabel="Dispersion k", ylabel="Generation reaching $threshold cases",
           title="(c) Outbreak speed | non-extinction  (R₀=$(R0_fig))",
           legend=false, size=(500, 300), margin=4Plots.mm,
           xticks=(1:length(k_vals), labels_c))

for (i, (k, col)) in enumerate(zip(k_vals, colors))
    data = results_c[k]
    isempty(data) && continue
    q25, q50, q75 = quantile(data, [0.25, 0.50, 0.75])
    iqr = q75 - q25
    lo  = max(minimum(data), q25 - 1.5*iqr)
    hi  = min(maximum(data), q75 + 1.5*iqr)
    pct = round(100 * length(data) / n_traj; digits=1)

    ## Box
    plot!(p2c, [i-0.3, i+0.3], [q25, q25]; color=col, lw=1.5, label=nothing)
    plot!(p2c, [i-0.3, i+0.3], [q75, q75]; color=col, lw=1.5, label=nothing)
    plot!(p2c, [i-0.3, i-0.3], [q25, q75]; color=col, lw=1.5, label=nothing)
    plot!(p2c, [i+0.3, i+0.3], [q25, q75]; color=col, lw=1.5, label=nothing)
    ## Median
    plot!(p2c, [i-0.3, i+0.3], [q50, q50]; color=col, lw=3,   label=nothing)
    ## Whiskers
    plot!(p2c, [i, i], [lo, q25]; color=col, lw=1, ls=:dash, label=nothing)
    plot!(p2c, [i, i], [q75, hi]; color=col, lw=1, ls=:dash, label=nothing)
    ## Fraction label
    annotate!(p2c, i, hi + 0.5, text("$(pct)%", 7, col))
end

plot(p2a, p2b, p2c; layout=(1,3), size=(1400, 320), margin=5Plots.mm)