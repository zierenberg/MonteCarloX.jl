# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # SIR with Heterogeneous Infectiousness (Lloyd-Smith et al., Nature 2005)
#
# Reproduction of Fig. 2a–b via a **contact process** (continuous-time Gillespie
# SIR) in the branching-process limit ``S \approx N \gg 1``.
#
# Each infector ``i`` draws an individual transmission rate
# ``\lambda_i \sim \mathrm{Gamma}(k,\, R_0\mu/k)`` on infection, so that
# ``\nu_i = \lambda_i/\mu \sim \mathrm{Gamma}(k, R_0/k)`` and the offspring
# count ``Z_i \sim \mathrm{NegBin}(R_0, k)`` in the large-``N`` limit.
#
# **Panel 2a**: Gamma distributions of ``\nu`` for varying ``k`` at ``R_0 = 1.5``.
#
# **Panel 2b**: simulated extinction probability ``q`` vs ``R_0``, validated
# against the analytical fixed-point ``q = g(q)`` of the pgf
# ``g(s) = (1 + R_0(1-s)/k)^{-k}``.
#
# Reference: Lloyd-Smith, J. O. et al. Nature 438, 355–359 (2005).

using Random, Distributions, Plots, Statistics, ProgressMeter
using MonteCarloX

# ## System definition
#
# The RNG is stored in the system so that `modify!` draws (random infector
# removal, new λ) use the same controlled stream as the Gillespie clock.

mutable struct SIRHetero <: AbstractSystem
    rng        :: Xoshiro
    lambdas    :: Vector{Float64}
    sum_lambda :: Float64
    mu         :: Float64
    P_lambda   :: Distribution
    rates      :: Vector{Float64}   # [recovery, infection]
    I          :: Int
end

function SIRHetero(rng, R0, k, mu; I0=1)
    P_lambda = k == Inf ? Dirac(R0 * mu) : Gamma(k, R0 * mu / k)
    lambdas  = rand(rng, P_lambda, I0)
    rates    = [mu * I0, sum(lambdas)]
    return SIRHetero(rng, lambdas, sum(lambdas), mu, P_lambda, rates, I0)
end

MonteCarloX.event_source(sys::SIRHetero) = sys.rates

function MonteCarloX.modify!(sys::SIRHetero, event::Int, t)
    if event == 1   # recovery: remove a random infector
        idx = rand(sys.rng, 1:sys.I)
        sys.sum_lambda -= sys.lambdas[idx]
        deleteat!(sys.lambdas, idx)
        sys.I -= 1
    else            # infection: new infector draws its own λ
        λ_new = rand(sys.rng, sys.P_lambda)
        push!(sys.lambdas, λ_new)
        sys.sum_lambda += λ_new
        sys.I += 1
    end
    ## S/N ≈ 1 in the branching-process limit: do not deplete susceptibles.
    ## Tracking S would reduce the effective R₀ as cases accumulate, biasing
    ## q upward relative to the analytical branching-process prediction.
    sys.rates[1] = sys.mu * sys.I
    sys.rates[2] = max(0.0, sys.sum_lambda)
    return nothing
end

# ## Simulation kernel
#
# Returns `true` if the outbreak went extinct before reaching `threshold` cases.
# Trajectories near R0=1 can be metastable — wandering between 0 and `threshold`
# for very long times. A wall-clock cutoff `T_max` resolves these: if time runs out,
# we check current `I` to decide (extinct iff I == 0).

function run_trajectory(rng, R0, k, mu, threshold; T_max=100_000.0)
    sys = SIRHetero(rng, R0, k, mu)
    alg = Gillespie(rng)
    total = 1
    while true
        t, event = step!(alg, sys.rates)
        alg.time > T_max && return sys.I == 0
        modify!(sys, event, t)
        event == 2 && (total += 1)
        total >= threshold && return false
    end
end

# ## Parameters

const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

mu        = 1/8
n_traj    = CI_MODE ? 200    : 2_000
threshold = 100

k_vals   = [0.1, 0.2, 0.5, 1.0, 2.0, Inf]
colors   = [:crimson, :orange, :gold, :steelblue, :mediumpurple, :black]
R0_sweep = CI_MODE ? collect(0.5:1.0:3.5) : collect(0.5:0.25:4.0)
R0_fig   = 1.5

# ## Analytical benchmark

function extinction_prob_analytical(R0, k)
    R0 <= 1 && return 1.0
    k == Inf && return exp(-(R0 - 1))
    q = 0.5
    for _ in 1:1000
        q_new = (1 + R0/k * (1 - q))^(-k)
        abs(q_new - q) < 1e-10 && return q_new
        q = q_new
    end
    return q
end

# ## Run simulations
#
# TODO: parallelise with the MCX parallel-chain pattern —
# one `Xoshiro` per thread, `Threads.@threads` over the (k, R0) grid.

q_sim = [zeros(length(R0_sweep)) for _ in k_vals]
prog  = Progress(length(k_vals) * length(R0_sweep); desc="Simulating: ", barlen=40)
for (ki, k) in enumerate(k_vals)
    for (ri, R0) in enumerate(R0_sweep)
        rng = Xoshiro(42 + ri)
        q_sim[ki][ri] = mean(run_trajectory(rng, R0, k, mu, threshold)
                             for _ in 1:n_traj)
        next!(prog)
    end
end
finish!(prog)

# ## Panel 2a: Gamma distributions of ν

x   = range(0, 6, length=300)
p2a = plot(xlabel="Individual reproductive number ν", ylabel="Probability density",
           title="(a) Distribution of ν  (R₀=$(R0_fig))",
           legend=:topright, size=(500, 320), margin=4Plots.mm)
for (k, col) in zip(k_vals, colors)
    label = k == Inf ? "k = ∞" : "k = $k"
    if k == Inf
        vline!(p2a, [R0_fig]; color=col, lw=2, label=label)
    else
        plot!(p2a, x, pdf.(Gamma(k, R0_fig/k), x); color=col, lw=2, label=label)
    end
end

# ## Panel 2b: extinction probability vs R0
# TODO: analytical curves are not correct, the limit case overlaps the nonlimit case!

p2b = plot(xlabel="R₀", ylabel="Extinction probability q",
           title="(b) Stochastic extinction",
           legend=:topright, ylims=(0,1),
           size=(500, 320), margin=4Plots.mm)
for (ki, (k, col)) in enumerate(zip(k_vals, colors))
    label = k == Inf ? "k = ∞" : "k = $k"
    scatter!(p2b, R0_sweep, q_sim[ki]; color=col, ms=4, alpha=0.7, label=nothing)
    R0_ana = range(0.5, 4.0, length=300)
    q_ana = [extinction_prob_analytical(R0, k) for R0 in R0_ana]
    plot!(p2b, R0_ana, q_ana; color=col, lw=2, label=label)
end
vline!(p2b, [1.0]; color=:gray, ls=:dash, lw=1, label=nothing)

plot(p2a, p2b; layout=(1,2), size=(1000, 340), margin=5Plots.mm)