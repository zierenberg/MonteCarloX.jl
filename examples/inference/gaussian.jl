# # Two Parameters — a Gaussian's Mean and Scale
#
# We observe data ``y_i`` and model them as ``y_i \sim \mathrm{Normal}(\mu, \sigma)``,
# inferring **both** the mean ``\mu`` and the scale ``\sigma > 0``. This is the step
# up from the [coin flip](coin_flip.md): two parameters instead of one, and one of
# them constrained to be positive. The same three sampling ideas apply — but here we
# see the first two start to run out of road, which is exactly why Markov chains (and
# later gradients) matter.
#
# Bayes' theorem is unchanged; ``\theta = (\mu,\sigma)`` is now a two-dimensional
# point, and the posterior ``p(\mu,\sigma \mid y)`` is a *surface* over the plane.

using Random, Distributions, StatsBase, Plots
using MonteCarloX

# ## Model, as plain functions of `θ = [μ, σ]`
#
# The prior is independent per parameter: a broad ``\mathrm{Normal}(0,10)`` on
# ``\mu`` and a ``\mathrm{LogNormal}(0,1)`` on ``\sigma`` (which is positive by
# construction, encoding the constraint). Anything with ``\sigma \le 0`` has zero
# posterior mass, which we express as a log-density of ``-\infty``.

rng = Xoshiro(1)
y   = rand(rng, Normal(3.0, 1.7), 200)          # truth: μ = 3.0, σ = 1.7

logprior(θ) = logpdf(Normal(0, 10), θ[1]) + logpdf(LogNormal(0, 1), θ[2])
loglik(θ)   = sum(logpdf.(Normal(θ[1], θ[2]), y))
logposterior(θ) = θ[2] > 0 ? logprior(θ) + loglik(θ) : -Inf
nothing #hide

# ## No conjugate shortcut
#
# Unlike the coin flip, this prior/likelihood pair is **not** conjugate: the
# posterior over ``(\mu,\sigma)`` is not a distribution we can name and sample from.
# So *simple sampling* — drawing from the posterior directly — is no longer on the
# table. To still get a ground truth for checking, we evaluate the (unnormalized)
# posterior on a fine grid and normalize it numerically; the evidence is literally
# the volume under that surface.

μg, σg = range(2.5, 3.5; length = 251), range(1.4, 2.1; length = 251)
logp   = [logposterior([μ, σ]) for σ in σg, μ in μg]
w      = exp.(logp .- maximum(logp))
cell   = step(μg) * step(σg)
log_evidence_grid = maximum(logp) + log(sum(w) * cell)
μ_grid = sum(w .* μg') / sum(w)                       # posterior mean of μ
σ_grid = sum(w .* σg)  / sum(w)                        # posterior mean of σ
(; μ_grid, σ_grid, log_evidence_grid)

# ## Importance sampling
#
# We can still draw from the prior and reweight to the posterior. It works — and
# again hands us the evidence for free — but notice the effective sample size: far
# below the coin flip's. In two dimensions the broad prior already places most of
# its draws where the data give them negligible weight. As the number of parameters
# grows this fraction shrinks toward zero, and importance sampling from the prior
# stops being usable. That decay is the practical reason we turn to Markov chains.

θs = [[rand(rng, Normal(0, 10)), rand(rng, LogNormal(0, 1))] for _ in 1:200_000]
iw = reweight(θs, logprior, logposterior)
(; μ_is = mean(first.(θs), weights(iw)), σ_is = mean(last.(θs), weights(iw)),
   log_evidence = log_normalization(iw), ess = round(Int, ess(iw)))

# ## Metropolis
#
# We define the move ourselves: one random-walk Metropolis update of the parameter
# vector — propose ``\theta' = \theta + \Delta\,\xi``, judge it with `accept!`, and move
# on acceptance (a proposal with ``\sigma' \le 0`` is auto-rejected). This is the
# continuous-parameter analogue of a spin flip: MonteCarloX supplies only the `accept!`
# judgement; the proposal is the model's to define.

function update!(θ, alg, Δ)
    θ′       = θ .+ Δ .* randn(alg.rng, length(θ))
    accepted = accept!(alg, θ′, θ)
    accepted && (θ .= θ′)
    return accepted
end
nothing #hide

# The run splits into two phases, like thermalization then production: a **warm-up** loop
# that adapts the step size ([`adapt!`](@ref) consumes each `update!`'s accept/reject to
# drive the acceptance toward 0.234), then a **sampling** loop with the step frozen.

function metropolis(logposterior; n = 100_000, warmup = 10_000, Δ0 = 1.0, seed = 1)
    rng  = Xoshiro(seed)
    alg  = MetropolisHastingsAlgorithm(rng, logposterior)
    step = AdaptiveStep(Δ0; target = 0.234)             # rough guess; self-corrects
    θ    = [mean(y), std(y)]

    for _ in 1:warmup                                   # warm-up: adapt the step size
        accepted = update!(θ, alg, step_size(step))
        adapt!(step, accepted)
    end
    reset!(alg)                                         # forget warm-up statistics

    Δ = step_size(step)                                 # freeze the step
    samples = zeros(2, n)
    for i in 1:n                                        # sampling
        update!(θ, alg, Δ)
        samples[:, i] = θ
    end
    return samples, alg
end
nothing #hide

samples, alg = metropolis(logposterior)
(; μ_mc = mean(samples[1, :]), σ_mc = mean(samples[2, :]),
   acceptance = round(acceptance_rate(alg); digits = 2))

# ## Comparison, and where this is heading
#
# Importance sampling and Metropolis both recover the grid posterior. But the story
# is the trend: *simple sampling* dropped out at two parameters, *importance
# sampling* is already wasting most of its draws, and only the Markov chain scales
# comfortably here. Push further — the [eight schools](eight_schools.md) model has
# ten parameters — and even random-walk Metropolis explores too slowly, because it
# diffuses. That is the point where **gradient-based** samplers (HMC), which use
# ``\nabla \log p`` to move purposefully, and the parameter transforms that free
# ``\sigma > 0`` into an unconstrained coordinate, start to pay off.

# Joint view: the Metropolis cloud fills the same region as the exact grid posterior
# (contours). Notice the truth sits slightly off-center — the posterior is built from
# *this* dataset, so it centers on the data's sample statistics (here sample std ≈ 1.82)
# rather than the true σ = 1.7. That gap is ordinary finite-sample fluctuation (about
# 1.4 sampling standard deviations for 200 points), *not* prior influence: the
# LogNormal(0,1) prior on σ is far too broad to matter against 200 observations.
# μ and σ are nearly independent here, so the cloud is roughly axis-aligned.
# Marginal of σ: grid reference (line) vs Metropolis (filled) — importance sampling
# agrees too (see the numbers above), but at ESS ≈ 200 its histogram is too ragged to plot.
pjoint = plot(xlabel = "μ", ylabel = "σ", title = "joint posterior p(μ, σ | y)", legend = :topright)
scatter!(pjoint, samples[1, 1:20:end], samples[2, 1:20:end]; ms = 1.5, alpha = 0.15,
         color = 3, label = "Metropolis")
contour!(pjoint, μg, σg, w; levels = 6, color = :black, colorbar = false, label = "grid posterior")
scatter!(pjoint, [3.0], [1.7]; ms = 7, color = :red, marker = :star5, label = "truth")

σmarg = vec(sum(w, dims = 2)); σmarg ./= sum(σmarg) * step(σg)
pσ = histogram(samples[2, :]; bins = 50, normalize = :pdf, alpha = 0.4, color = 3,
               label = "Metropolis", xlabel = "σ", ylabel = "density", title = "marginal of σ")
plot!(pσ, σg, σmarg; lw = 3, color = :black, label = "grid")

plot(pjoint, pσ; layout = (1, 2), size = (950, 360), margin = 4Plots.mm)
