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
    isfinite(τ) || return -Inf     # overflow guard: extreme proposals get zero mass, not an error
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

# Same two-phase shape as the other examples, but here our `update!` is *component-wise*
# — it moves one parameter at a time, which mixes far better than a joint move in ten
# dimensions. The [`AdaptiveStep`](@ref) carries per-component sizes (its ratios) and
# adapts their overall magnitude to the 0.234 target during warm-up.

function update!(s, alg, Δ)
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
        accepted = update!(s, alg, step_size(step))
        adapt!(step, accepted)
    end
    reset!(alg)

    Δ = step_size(step)                                      # freeze
    samples = zeros(length(s), n)
    for i in 1:n                                             # sampling
        update!(s, alg, Δ)
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

# ## Gradient-based sampling: Hamiltonian Monte Carlo
#
# Ten dimensions is where the random walk starts to hurt. Component-wise Metropolis
# accepts often, but each accepted move shifts a single coordinate a little — the chain
# *diffuses*, and successive samples stay correlated for a long time. **Hamiltonian
# Monte Carlo** (HMC) proposes differently: augment the parameters ``s`` with a momentum
# ``p \sim \mathcal{N}(0, I)`` and treat
#
# ```math
# H(s, p) = -\log p(s \mid y) + \tfrac12 \lVert p \rVert^2
# ```
#
# as an energy. Integrating the corresponding Hamiltonian dynamics (``L`` *leapfrog*
# steps of size ``\epsilon``) carries the state far across parameter space while nearly
# conserving ``H`` — a *distant* proposal that is still likely to be accepted. The
# integrator needs ``\nabla \log p``, and that is the only genuinely new ingredient;
# every piece of the machinery is one we already have:
#
# - **The gradient comes from the ecosystem.** A [`FunctionEnsemble`](@ref) with its
#   `dimension` declared satisfies the LogDensityProblems.jl interface, so an AD backend
#   (here ForwardDiff) equips the *same* log-posterior with a gradient — no derivative
#   code in the model, none in MonteCarloX.
# - **The accept step is plain Metropolis.** Leapfrog conserves ``H`` only up to
#   integration error; accepting with probability ``\min(1, e^{H_0 - H})`` corrects it
#   exactly. That is a [`Metropolis`](@ref) judgement at ``\beta = 1`` on the energy
#   ``H`` — the same `accept!` that judged the spin flip and the random walk.
# - **The step size is adapted the same way.** The same [`AdaptiveStep`](@ref) drives
#   ``\epsilon`` toward HMC's optimal acceptance rate of ``\approx 0.65``.

using LogDensityProblems, LogDensityProblemsAD, ForwardDiff

ℓ = ADgradient(:ForwardDiff, FunctionEnsemble(logposterior; dimension = 2 + J))
∇logp(s) = LogDensityProblems.logdensity_and_gradient(ℓ, s)

# The proposal — the leapfrog flight — is ours to define, exactly like the
# component-wise move before it:

function hmc_update!(s, alg, ϵ; L = 20)
    p     = randn(alg.rng, length(s))                # fresh momentum
    lp, g = ∇logp(s)
    H0    = -lp + 0.5 * sum(abs2, p)
    s′    = copy(s)
    for _ in 1:L
        p  .+= 0.5ϵ .* g                             # half momentum kick
        s′ .+= ϵ .* p                                # full position drift
        lp, g = ∇logp(s′)
        p  .+= 0.5ϵ .* g                             # half momentum kick
    end
    H = -lp + 0.5 * sum(abs2, p)
    accepted = accept!(alg, H - H0)                  # Metropolis at β = 1 on H
    accepted && (s .= s′)
    return accepted
end

# The driver has the same two-phase shape as every run in this series — warm-up with
# adaptation, freeze, sample:

function hmc(logposterior, s0; n = 2_000, warmup = 500, seed = 42)
    rng  = Xoshiro(seed)
    alg  = Metropolis(rng; β = 1.0)                  # judges ΔH, an energy difference
    step = AdaptiveStep(0.1; target = 0.65)
    s    = copy(s0)

    for _ in 1:warmup                                # warm-up: adapt the leapfrog step
        accepted = hmc_update!(s, alg, step_size(step))
        adapt!(step, accepted)
    end
    reset!(alg)

    ϵ = step_size(step)                              # freeze
    samples = zeros(length(s), n)
    for i in 1:n                                     # sampling
        hmc_update!(s, alg, ϵ)
        samples[:, i] = s
    end
    return samples, alg
end

samples_hmc, alg_hmc = hmc(logposterior, s0)
μ_hmc = samples_hmc[1, :]
τ_hmc = exp.(samples_hmc[2, :])
(; μ = round(mean(μ_hmc), digits = 1), τ = round(median(τ_hmc), digits = 1),
   acceptance = round(acceptance_rate(alg_hmc); digits = 2))

# Same posterior — but count the *independent* draws. The autocorrelation-based
# [`ess`](@ref) makes the difference concrete (each HMC sample costs ``L + 1 = 21``
# gradient evaluations, so compare information per posterior evaluation, not per sample):

(; ess_metropolis = round(Int, ess(μ_post)), n_metropolis = length(μ_post),
   ess_hmc = round(Int, ess(μ_hmc)), n_hmc = length(μ_hmc))

# ## Where this is heading
#
# The HMC above is deliberately minimal — fixed path length, unit mass matrix, a single
# chain — because its point is the *composition*: a caller-owned proposal (leapfrog),
# the universal `accept!` judgement, `AdaptiveStep` for tuning, and a gradient borrowed
# through the LogDensityProblems seam. The production route keeps the target exactly as
# it is and swaps only the driver: AdvancedHMC.jl's NUTS adds dynamic path lengths,
# dual-averaging step sizes, and mass-matrix adaptation through the same bridge.
#
# One honest caveat remains. This centered parameterization has a notorious difficulty:
# when ``\tau`` is small the effects ``\theta_j`` are squeezed tightly around ``\mu``,
# forming a *funnel* that both the random walk and basic HMC explore poorly. The
# standard fix is the *non-centered* reparameterization ``\theta_j = \mu + \tau\,z_j``
# with ``z_j \sim \mathcal{N}(0,1)`` — a change of the *target*, not the sampler.
# Transforms, gradients, and drivers around a log-density are generic; spelling them out
# by hand here marks exactly the seam where a future MCXInference layer would generalize.
