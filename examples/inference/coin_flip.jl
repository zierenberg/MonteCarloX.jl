# # Bayesian Inference by Sampling — the Coin Flip
#
# We flip a coin `n` times and see `h` heads. What is its bias
# ``\theta = P(\text{heads})``? *Bayesian* inference does not return a single
# number; it returns a whole probability distribution over ``\theta`` that encodes
# what the data taught us and how uncertain we still are.
#
# ## The four ingredients
#
# Everything follows from **Bayes' theorem**:
#
# ```math
# \underbrace{p(\theta \mid \text{data})}_{\text{posterior}}
#   = \frac{\overbrace{p(\text{data}\mid\theta)}^{\text{likelihood}}\;
#           \overbrace{p(\theta)}^{\text{prior}}}
#          {\underbrace{p(\text{data})}_{\text{evidence}}}
# ```
#   
# - **prior** ``p(\theta)`` — what we believe about ``\theta`` *before* seeing data.
# - **likelihood** ``p(\text{data}\mid\theta)`` — how probable the observed data is
#   for a given ``\theta`` (as a function of ``\theta``, not of the data).
# - **posterior** ``p(\theta\mid\text{data})`` — the updated belief. This is the goal.
# - **evidence** ``p(\text{data}) = \int p(\text{data}\mid\theta)\,p(\theta)\,d\theta`` —
#   the normalizing constant. It is an integral over *all* ``\theta`` and is usually
#   the hard part: in realistic models it cannot be computed in closed form.
#
# The whole difficulty of Bayesian computation is that we can write down the
# *numerator* (prior × likelihood) easily, but not the *denominator* (evidence).
# Sampling methods are the way out. Three of them — mirroring the simple /
# importance / rejection trichotomy of ordinary Monte Carlo — already cover a lot,
# and the coin flip lets us see all three side by side against the exact answer.

using Random, Distributions, StatsBase, Plots
using MonteCarloX

# ## The model, written as plain functions
#
# We keep the three ingredients as functions of ``\theta`` — no special types.
# The prior is a ``\text{Beta}(2,2)`` (a mild belief that the coin is roughly fair);
# the likelihood of `h` heads in `n` flips is the binomial ``\theta^h(1-\theta)^{n-h}``
# (we drop the constant binomial coefficient — it cancels everywhere that matters).

data = (heads = 61, flips = 100)

logprior(θ)     = logpdf(Beta(2, 2), θ)
loglik(θ)       = data.heads * log(θ) + (data.flips - data.heads) * log1p(-θ)
logposterior(θ) = logprior(θ) + loglik(θ)       # numerator only — unnormalized

# This particular prior/likelihood pair is *conjugate*: the posterior is again a
# Beta distribution, ``\text{Beta}(2+h,\,2+(n-h))``. That gives us an exact answer
# to check the samplers against — a luxury we lose in every harder model.

posterior_exact = Beta(2 + data.heads, 2 + data.flips - data.heads)

# ## 1. Simple sampling
#
# When the posterior is a distribution we recognize (as here, thanks to conjugacy),
# there is nothing to do but draw from it directly. No Markov chain, no weights.
# This is the Bayesian version of "the source *is* the target": the honest lesson
# is that simple models do not need heavy machinery.

draws_simple = rand(posterior_exact, 100_000)
(; method = "simple", mean = mean(draws_simple))

# ## 2. Importance sampling
#
# Usually we cannot sample the posterior directly, but we *can* sample the prior.
# Importance sampling draws ``\theta`` from the prior and **reweights** each draw by
# how well it explains the data (its likelihood). [`reweight`](@ref) forms those
# weights in log space; `weights(iw)` turns them into posterior-weighted samples.

θs = rand(Beta(2, 2), 100_000)
iw = reweight(θs, logprior, logposterior)      # source = prior, target = posterior
(; method = "importance", mean = mean(θs, weights(iw)), ess = round(Int, ess(iw)))

# The bonus is the **evidence**. Because the weights are exactly
# ``p(\text{data}\mid\theta)`` and the prior is normalized,
# ``\log`` of their average is the log-evidence — the one quantity direct posterior
# sampling and Metropolis (below) cannot give us. Evidence is an integral of
# prior × likelihood, so we can check it by brute-force quadrature:

θgrid    = range(0, 1; length = 20_000)[2:end-1]
evidence = sum(pdf.(Beta(2, 2), θgrid) .* exp.(loglik.(θgrid))) * step(θgrid)
(; log_evidence_is = log_normalization(iw), log_evidence_exact = log(evidence))

# Importance sampling is cheap and gives the evidence for free, but it only works
# while the prior and posterior overlap well. As data become more informative (or
# parameters more numerous) the posterior concentrates in a tiny corner of the
# prior, almost all weights collapse to zero, and `ess` drops — the signal that we
# need a method that actively seeks out the posterior. That is Metropolis.

# ## 3. Metropolis (Markov chain Monte Carlo)
#
# Metropolis builds a random walk whose stationary distribution *is* the posterior.
# From the current ``\theta`` it proposes a nearby ``\theta'`` and accepts it with
# probability ``\min(1, p(\theta'\mid\text{data})/p(\theta\mid\text{data}))`` — a
# ratio in which the intractable evidence **cancels**. So Metropolis needs only the
# unnormalized posterior (prior × likelihood), which is exactly what we can write
# down. `accept!` evaluates the log-ratio and tracks the acceptance statistics.

# We define our own move, as each example does. Here ``\theta`` is a single scalar
# bounded to ``(0,1)`` (out-of-range proposals are rejected). A scalar cannot be mutated
# in place, so this `update!` returns the (possibly new) ``\theta`` and whether it moved.

function update!(θ, alg, Δ)
    θ′       = θ + Δ * randn(alg.rng)
    accepted = 0.0 < θ′ < 1.0 && accept!(alg, θ′, θ)
    return (accepted ? θ′ : θ), accepted
end

# Same two-phase shape as the other examples — a warm-up loop that adapts the step size,
# then a frozen sampling loop; the 1-D acceptance target is the higher `0.44`.

function metropolis(logposterior; n = 100_000, warmup = 10_000, Δ0 = 0.2, seed = 1)
    rng  = Xoshiro(seed)
    alg  = MetropolisHastingsAlgorithm(rng, logposterior)
    step = AdaptiveStep(Δ0; target = 0.44)
    θ    = 0.5

    for _ in 1:warmup                                       # warm-up: adapt the step size
        θ, accepted = update!(θ, alg, step_size(step))
        adapt!(step, accepted)
    end
    reset!(alg)

    Δ = step_size(step)                                     # freeze the step
    samples = Float64[]
    for _ in 1:n                                            # sampling
        θ, _ = update!(θ, alg, Δ)
        push!(samples, θ)
    end
    return samples, alg
end

draws_mcmc, alg = metropolis(logposterior)
(; method = "metropolis", mean = mean(draws_mcmc), acceptance = round(acceptance_rate(alg); digits = 2))

# ## All three vs. the exact posterior
#
# Each method recovers the same Beta posterior — simple sampling by construction,
# importance sampling by reweighting the prior, Metropolis by exploring. The three
# histograms sit on top of the exact curve, and the prior shows how far the data
# moved our belief.

xgrid = range(0.0, 1.0; length = 300)
plt = plot(xgrid, pdf.(posterior_exact, xgrid); lw = 3, color = :black,
           label = "exact posterior", xlabel = "θ", ylabel = "density",
           title = "Coin flip: prior → posterior")
plot!(plt, xgrid, pdf.(Beta(2, 2), xgrid); lw = 2, ls = :dash, color = :gray, label = "prior")
stephist!(plt, draws_simple; bins = 60, normalize = :pdf, lw = 2, color = 1, label = "simple")
stephist!(plt, θs; weights = weights(iw), bins = 60, normalize = :pdf, lw = 2, color = 2, label = "importance")
stephist!(plt, draws_mcmc; bins = 60, normalize = :pdf, lw = 2, color = 3, label = "metropolis")
plt
