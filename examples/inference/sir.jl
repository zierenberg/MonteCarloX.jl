# # Dynamical Models — Inferring Epidemic Rates (SIR)
#
# The previous examples had likelihoods we could write in closed form. Here the model
# is a *mechanism*: a system of differential equations describing how an epidemic
# spreads. We observe noisy case counts over time and infer the rates that drive the
# dynamics. What is new is that **evaluating the likelihood runs a simulation** — every
# proposed parameter requires solving the ODE — so the target is expensive but
# otherwise just another `logposterior` function.
#
# The SIR model splits a population into Susceptible, Infected, and Recovered
# fractions, with a transmission rate ``\beta`` and a recovery rate ``\gamma``:
#
# ```math
# \dot S = -\beta S I, \qquad \dot I = \beta S I - \gamma I, \qquad \dot R = \gamma I.
# ```

using Random, Distributions, StatsBase, Plots, DelimitedFiles
using OrdinaryDiffEq
using MonteCarloX

datadir      = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "data")))  # hide
samples_file = joinpath(datadir, "sir_samples.tsv")   # hide
meta_file    = joinpath(datadir, "sir_meta.tsv")      # hide

function sir!(du, u, θ, t)
    β, γ = θ
    S, I, R = u
    du[1] = -β * S * I
    du[2] =  β * S * I - γ * I
    du[3] =  γ * I
end

u0    = [0.99, 0.01, 0.0]      # initial S, I, R fractions
tspan = (0.0, 15.0)
tobs  = 0.0:0.5:15.0

# Solve the ODE for rates `θ` and return the infected curve at the observation times.
solve_I(θ) = solve(ODEProblem(sir!, u0, tspan, θ), Tsit5(); saveat = tobs)[2, :]

# ## Synthetic data
#
# Real epidemic data are *counts*, not fractions, so we model observed cases at each
# time as ``C(t) \sim \text{Poisson}(N\,I(t))`` for a population ``N``. This is the
# right noise model here: its spread scales with the signal, so the small values at
# the start and in the recovery tail are not swamped (as constant additive Gaussian
# noise would swamp them) — and since ``\gamma`` is inferred from that decaying tail,
# using the correct observation model is what lets us recover it without bias.

rng   = Xoshiro(1)
truth = [1.5, 0.4]                          # β = 1.5, γ = 0.4  (basic reproduction R₀ ≈ 3.75)
N     = 1000                                # population size
data  = rand.(rng, Poisson.(N .* solve_I(truth)))

# ## Model
#
# Positive rates get log-normal priors. The likelihood solves the ODE at `θ`, turns
# the infected fractions into expected counts ``N\,I(t)``, and scores the observed
# counts under a Poisson; a non-positive rate has no posterior mass.

logprior(θ) = logpdf(LogNormal(log(1.5), 0.5), θ[1]) + logpdf(LogNormal(log(0.5), 0.5), θ[2])
loglik(θ)   = sum(logpdf.(Poisson.(N .* max.(solve_I(θ), 1e-12)), data))
logposterior(θ) = all(θ .> 0) ? logprior(θ) + loglik(θ) : -Inf

# ## Metropolis
#
# The step per iteration is an ODE solve, but the sampler is unchanged. The move is the
# same random-walk `update!` as the [Gaussian](gaussian.md) — the model defines it, only
# `accept!` comes from MonteCarloX — and the run is the same two-phase structure: a
# warm-up loop that adapts the proposal size ([`AdaptiveStep`](@ref) + [`adapt!`](@ref)),
# then a sampling loop with the step frozen.

function update!(θ, alg, Δ)
    θ′       = θ .+ Δ .* randn(alg.rng, length(θ))
    accepted = accept!(alg, θ′, θ)
    accepted && (θ .= θ′)
    return accepted
end

function metropolis(logposterior; n = 10_000, warmup = 2_000, Δ0 = [0.1, 0.04], seed = 1)
    rng  = Xoshiro(seed)
    alg  = MarkovChainMonteCarlo(rng, logposterior)
    step = AdaptiveStep(Δ0; target = 0.234)
    θ    = [1.5, 0.5]

    for _ in 1:warmup                                   # warm-up: adapt the step size
        accepted = update!(θ, alg, step_size(step))
        adapt!(step, accepted)
    end
    reset!(alg)

    Δ = step_size(step)                                 # freeze the step
    samples = zeros(2, n)
    for i in 1:n                                        # sampling
        update!(θ, alg, Δ)
        samples[:, i] = θ
    end
    return samples, alg
end

if !isfile(samples_file)                                       # hide
samples, alg = metropolis(logposterior)
mkpath(datadir)                                                # hide
writedlm(samples_file, ["beta" "gamma"; permutedims(samples)], '\t')  # hide
writedlm(meta_file, ["acceptance"; acceptance_rate(alg)], '\t')       # hide
end                                                            # hide
samples = permutedims(readdlm(samples_file, '\t'; header = true)[1])   # hide
acc     = readdlm(meta_file, '\t'; header = true)[1][1]                # hide
(; β = round(mean(samples[1, :]), digits = 2), γ = round(mean(samples[2, :]), digits = 2),
   truth = truth, acceptance = round(acc; digits = 2))

# ## Results
#
# The posterior concentrates around the true rates, and the trajectory implied by the
# posterior-mean rates tracks the data. The right panel shows the joint posterior over
# ``(\beta, \gamma)`` with the truth marked — the two rates are correlated, since a
# faster-spreading, faster-recovering epidemic can produce a similar curve.

β̂, γ̂ = mean(samples[1, :]), mean(samples[2, :])

pfit = plot(tobs, N .* solve_I(truth); lw = 2, color = :black, ls = :dash, label = "truth",
            xlabel = "time", ylabel = "infected count", title = "fit")
scatter!(pfit, tobs, data; ms = 3, color = 1, label = "data (counts)")
plot!(pfit, tobs, N .* solve_I([β̂, γ̂]); lw = 3, color = 3, label = "posterior mean")

ppost = scatter(samples[1, :], samples[2, :]; ms = 1, alpha = 0.2, color = 3,
                xlabel = "β", ylabel = "γ", title = "joint posterior", label = "")
scatter!(ppost, [truth[1]], [truth[2]]; ms = 6, color = :black, marker = :star5, label = "truth")

plot(pfit, ppost; layout = (1, 2), size = (900, 340), margin = 4Plots.mm)

# ## Beyond a deterministic model
#
# Here the simulator is deterministic, so the likelihood is tractable. When the
# dynamics are *stochastic* — e.g. a Gillespie SIR with small populations — there is no
# closed-form likelihood at all. Inference then switches to **ABC** (approximate
# Bayesian computation): keep parameter draws whose *simulated* data resemble the
# observed data under some distance, reusing MonteCarloX's Gillespie tools as the
# simulator. Same posterior target, a likelihood replaced by a simulate-and-compare step.
