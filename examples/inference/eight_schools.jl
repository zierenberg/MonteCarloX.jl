# # Hierarchical Models — the Eight Schools
#
# A coaching program was tested in eight schools. Each school reports an estimated
# effect ``y_j`` with a known standard error ``\sigma_j`` — but the estimates are
# noisy and scattered. How much should we believe any single school?
#
# Two extremes are both wrong. *No pooling* trusts each ``y_j`` on its own and
# overfits the noise. *Complete pooling* collapses all schools to one number and
# ignores real differences. The Bayesian answer is **partial pooling**: assume the
# schools' true effects ``\theta_j`` are themselves drawn from a shared population
# distribution, and *learn that distribution from the data*.
#
# ```math
# \mu,\ \tau \sim \text{priors}, \qquad
# \theta_j \sim \mathcal{N}(\mu, \tau), \qquad
# y_j \sim \mathcal{N}(\theta_j, \sigma_j)
# ```
#
# This is *hierarchical*: the parameters ``\theta_j`` are governed by
# **hyperparameters** ``(\mu, \tau)`` that are themselves inferred. That extra layer
# is what couples the schools and lets each one borrow strength from the others.

using Random, Distributions, StatsBase, Plots
using MonteCarloX

y = [28.0,  8.0, -3.0,  7.0, -1.0,  1.0, 18.0, 12.0]     # estimated effects
σ = [15.0, 10.0, 16.0, 11.0,  9.0, 11.0, 10.0, 18.0]     # their standard errors
J = length(y)

# ## The model as one log-posterior over a flat state
#
# The unknowns are the population mean ``\mu``, the population spread ``\tau > 0``,
# and the eight effects ``\theta_j`` — ten parameters, stacked into one vector
# ``s = [\mu,\ \log\tau,\ \theta_1,\dots,\theta_8]``. We sample ``\log\tau`` rather
# than ``\tau`` so the chain moves in an unconstrained space; changing variable from
# ``\tau`` to ``\log\tau`` multiplies the density by ``\tau``, i.e. adds ``\log\tau``
# to the log-prior (a *Jacobian* term — the same bookkeeping a transform layer would
# automate). The prior on ``\tau`` is the standard weakly-informative half-Cauchy.

function logposterior(s)
    μ, logτ = s[1], s[2]
    θ = @view s[3:end]
    τ = exp(logτ)
    lp  = logpdf(Normal(0, 10), μ)                              # hyperprior on μ
    lp += logpdf(truncated(Cauchy(0, 5), 0, Inf), τ) + logτ    # half-Cauchy on τ, + Jacobian
    lp += sum(logpdf.(Normal(μ, τ), θ))                         # population → effects
    lp += sum(logpdf.(Normal.(θ, σ), y))                        # effects → data
    return lp
end

# ## Only MCMC survives here
#
# There is no conjugate form to sample directly, so *simple sampling* is out. And in
# ten dimensions, drawing from the prior and reweighting would give an effective
# sample size of essentially zero (the two-parameter [Gaussian](gaussian.md) already
# fell to ESS/N ≈ 0.001) — so *importance sampling* is out too. A Markov chain that
# actively explores the posterior is the only workable option.

# Same two-phase shape as the other examples, but in ten dimensions a *component-wise*
# random walk (one parameter at a time) mixes far better than a joint move, so the step
# is written out rather than using the vector `update!`. The [`AdaptiveStep`](@ref)
# carries per-component sizes (its ratios) and adapts their overall magnitude to the
# 0.234 target during warm-up.

function step_component!(s, alg, Δ)
    k     = rand(alg.rng, 1:length(s))                       # update one component
    s_new = copy(s); s_new[k] += Δ[k] * randn(alg.rng)
    accepted = accept!(alg, s_new, s)
    accepted && (s[k] = s_new[k])
    return accepted
end

function metropolis(logposterior, s0; n = 200_000, warmup = 20_000, seed = 42)
    rng  = Xoshiro(seed)
    alg  = MarkovChainMonteCarlo(rng, logposterior)
    step = AdaptiveStep([2.0, 0.4, fill(6.0, length(s0) - 2)...]; target = 0.234)
    s    = copy(s0)

    for _ in 1:warmup                                        # warm-up: adapt the step sizes
        accepted = step_component!(s, alg, step_size(step))
        adapt!(step, accepted)
    end
    reset!(alg)

    Δ = step_size(step)                                      # freeze
    samples = zeros(length(s), n)
    for i in 1:n                                             # sampling
        step_component!(s, alg, Δ)
        samples[:, i] = s
    end
    return samples, alg
end

s0 = [mean(y); log(std(y)); copy(y)]
samples, alg = metropolis(logposterior, s0)

μ_post = samples[1, :]
τ_post = exp.(samples[2, :])
θ_post = samples[3:end, :]
(; μ = round(mean(μ_post), digits = 1), τ = round(median(τ_post), digits = 1),
   acceptance = round(acceptance_rate(alg); digits = 2))

# ## Partial pooling, made visible
#
# The point of the hierarchy is *shrinkage*: each school's posterior effect is pulled
# from its noisy raw estimate ``y_j`` toward the population mean ``\mu``, by an amount
# set by how noisy that school is. Schools with large ``\sigma_j`` (little
# information) are pulled the most. This is the model borrowing strength across
# schools — impossible without the shared population layer.

μ̂ = mean(μ_post)
θ̂ = vec(mean(θ_post, dims = 2))

plt = plot(xlabel = "school", ylabel = "effect", title = "shrinkage toward the population mean",
           legend = :topright, xticks = 1:J)
hline!(plt, [μ̂]; color = :gray, ls = :dash, label = "population mean μ ≈ $(round(μ̂, digits=1))")
scatter!(plt, 1:J, y; yerror = σ, ms = 5, color = 1, label = "raw estimate yⱼ ± σⱼ")
scatter!(plt, 1:J, θ̂; ms = 5, color = 3, label = "posterior effect θⱼ")

# ## Per-school posterior distributions
#
# The shrinkage above is a summary; the full story is each school's *posterior
# distribution* for its effect ``\theta_j``. Each panel shows that distribution with
# the school's raw estimate ``y_j`` (solid) and the population mean ``\mu`` (dashed)
# marked — every posterior is pulled from its noisy raw value toward the population,
# and all eight are broad and heavily overlapping, reflecting how little eight schools
# pin down any individual effect.

panels = map(1:J) do j
    p = histogram(θ_post[j, :]; bins = 40, normalize = :pdf, alpha = 0.5, color = 3,
                  legend = false, title = "school $(j)", xlims = (-12, 32))
    vline!(p, [y[j]]; color = 1, lw = 2)                 # raw estimate yⱼ
    vline!(p, [μ̂];   color = :gray, ls = :dash, lw = 2)  # population mean μ
    p
end
plot(panels...; layout = (2, 4), size = (1100, 450), margin = 3Plots.mm)

# ## Where this is heading
#
# This centered parameterization has a notorious difficulty: when ``\tau`` is small
# the effects ``\theta_j`` are squeezed tightly around ``\mu``, forming a *funnel*
# that a random walk (and even basic HMC) struggles to explore. The standard fix is a
# *non-centered* reparameterization, ``\theta_j = \mu + \tau\,z_j`` with
# ``z_j \sim \mathcal{N}(0,1)`` — a transform of the parameters. Combined with
# gradient-based sampling (HMC), that is what makes genuinely high-dimensional
# hierarchical models tractable, and it is the direction the inference API is built toward.
