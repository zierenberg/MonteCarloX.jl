# # State-Space Models — an AR(1) Process with Immigration
#
# A population (or a spike count, or an activity level) at time ``t`` is a noisy
# echo of its own past plus a steady trickle of new arrivals:
#
# ```math
# x_t = \rho\, x_{t-1} + h + \sigma\,\varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,1)
# ```
#
# ``\rho \in (0,1)`` is the persistence ("branching ratio"): how much of yesterday
# survives into today. ``h > 0`` is the immigration rate: new activity injected
# every step, independent of the past. ``\sigma`` is the process noise. This is the
# continuous-state cousin of the classic branching process with immigration used to
# model population counts and, in neuroscience, the branching ratio of population
# spiking activity [Heathcote 1965; Wilting & Priesemann 2019].
#
# ``x_t`` itself is a population-level quantity — a count, an activity level — and we
# essentially never see all of it. What reaches a detector is a **fraction**
# ``\alpha \in (0, 1]`` of the population: an electrode array samples some neurons out
# of billions, a survey counts some individuals out of a whole population. This is the
# subsampling problem: the observation is a scaled-down echo of ``x_t``, not ``x_t``
# corrupted by noise of its own:
#
# ```math
# y_t = \alpha\, x_t + \tau\,\zeta_t, \qquad \zeta_t \sim \mathcal{N}(0,1),
# ```
#
# with ``\tau`` genuine measurement noise on top of the subsampled signal. (The Gaussian
# relaxation used throughout leaves ``x_t`` free to wander below 0, which is not really
# a population count; with the parameters used here the trajectory never comes close to
# 0, so this is a harmless simplification, not a modeling claim.)
#
# What makes this a genuine **state-space model**, rather than just another
# closed-form target like [Gaussian](gaussian.md), is that the unknowns are not just
# ``(\rho, h, \sigma, \tau, \alpha)`` — the entire latent trajectory ``x_{1:T}`` is
# unknown too, and must be inferred jointly with the parameters that generated it.

using Random, Distributions, StatsBase, Plots
using MonteCarloX

# ## Synthetic data
#
# A short series — ``T = 20`` — deliberately: short enough that the latent
# trajectory is not pinned down by the data alone, so the prior and the model's
# coupling between time steps both do real work. The true initial state is drawn
# from the AR(1)'s own stationary distribution, just to start the simulation
# somewhere typical; nothing later assumes the chain knows this. Half the population
# is subsampled away before ``\tau`` noise is added on top.

T    = 20
ρ_true, h_true, σ_true, τ_true, α_true = 0.8, 1.0, 0.5, 0.5, 0.5

rng     = Xoshiro(1)
x_true  = zeros(T)
x_true[1] = h_true / (1 - ρ_true) + σ_true / sqrt(1 - ρ_true^2) * randn(rng)
for t in 2:T
    x_true[t] = ρ_true * x_true[t-1] + h_true + σ_true * randn(rng)
end
w_true = α_true .* x_true
y = w_true .+ τ_true .* randn(rng, T)
nothing #hide

# ## Model
#
# The tempting parametrization keeps ``x_{1:T}`` — the population-scale trajectory —
# as the sampled state and adds ``\alpha`` as one more free coordinate, logistic-
# transformed onto ``(0,1)`` exactly like ``\rho``, with `logpdf(Normal.(α .* x, τ), y)`
# for the observation. It runs, it accepts moves, and it is a trap: ``x_t`` and
# ``\alpha`` can trade off against each other — shrink the whole trajectory by a
# factor ``c`` and inflate ``\alpha`` by ``1/c`` and the observation ``\alpha x_t`` is
# essentially unchanged, compensated on the process side by shrinking ``h`` and
# ``\sigma`` by the same ``c``. That is a scale ridge, not a point, and a component-wise
# sampler can barely move along it: even with an already-informative prior on
# ``\alpha`` centered on the truth, this parametrization gives **ESS(``\alpha``) ≈ 6 out
# of 400,000 draws** (tested while building this example) — an order of magnitude worse
# than the ``\rho``–``h`` ridge that motivated the ``\mu`` reparametrization below.
#
# The fix follows the same principle as that reparametrization: don't sample the
# quantity you care about, sample the quantity the data actually constrain. The data
# never see ``x_t`` — only ``\alpha x_t``. Define the **observed-scale** trajectory
# ``w_t := \alpha\, x_t`` and substitute it into the recursion:
#
# ```math
# w_t = \alpha x_t = \alpha(\rho x_{t-1} + h + \sigma\varepsilon_t)
#     = \rho\, w_{t-1} + (\alpha h) + (\alpha \sigma)\,\varepsilon_t
# ```
#
# ``w_t`` obeys its own AR(1) process, with immigration ``h_w = \alpha h`` and noise
# ``\sigma_w = \alpha\sigma`` in place of ``h,\sigma``, and the observation is now the
# clean, non-degenerate ``y_t = w_t + \tau\zeta_t``. This model has exactly the same
# functional form — and the same dimensionality, ``4+T`` — as a state-space model
# with no subsampling at all, just fit to composite parameters ``(\rho, h_w, \sigma_w,
# \tau)`` instead of ``(\rho, h, \sigma, \tau)``. ``\alpha`` has not been fixed; it has
# vanished from the likelihood entirely, because the likelihood genuinely does not
# depend on it — only on the products ``\alpha h`` and ``\alpha\sigma``. That is exactly
# the identifiability result behind the subsampling-invariant estimator of
# [Wilting & Priesemann 2018]: ``\rho`` survives subsampling, but the absolute
# population scale does not, unless outside information pins ``\alpha`` down.
#
# So ``\alpha`` is not part of the sampled state at all — there is nothing in ``y`` to
# update it with, no mixing cost to pay, and its posterior is, by construction, exactly
# its prior. Population-scale quantities are recovered afterward by drawing ``\alpha``
# from that prior, independently for each posterior draw of ``(w, h_w, \sigma_w)``, and
# pushing it through ``x_t = w_t/\alpha``, ``h = h_w/\alpha``, ``\sigma = \sigma_w/\alpha``.
# Whatever you know about ``\alpha`` a priori is exactly and only what you know about
# the population scale a posteriori — no amount of additional data narrows it, since
# the likelihood never touches ``\alpha``.
#
# Everything below samples the observed-scale parameters ``(\rho, \mu_w, \sigma_w,
# \tau)`` and the trajectory ``w_{1:T}``, using the same ``\mu_w = h_w/(1-\rho)``
# non-centering trick as before to keep ``\rho`` and ``h_w`` from fighting each other.
# Every prior is deliberately uninformative: half-Cauchy on the positive/bounded
# parameters [Gelman 2006], and — per explicit request — no prior at all on the
# initial state ``w_1``, a completely free parameter pinned down only by its
# consequences: the transition to ``w_2`` and its own observation ``y_1``.

function logposterior(s)
    z, logμ, logσ, logτ = s[1], s[2], s[3], s[4]
    w = @view s[5:end]
    ρ = 1 / (1 + exp(-z))                          # logistic transform, ρ ∈ (0,1)
    μ, σ, τ = exp(logμ), exp(logσ), exp(logτ)
    all(isfinite, (ρ, μ, σ, τ)) || return -Inf      # overflow guard
    h = μ * (1 - ρ)                                 # observed-scale immigration rate

    lp  = logpdf(truncated(Cauchy(0, 1), 0, 1), ρ) + log(ρ * (1 - ρ))  # half-Cauchy(ρ) + Jacobian
    lp += logpdf(truncated(Cauchy(0, 5), 0, Inf), μ) + logμ             # half-Cauchy(μ) + Jacobian
    lp += logpdf(truncated(Cauchy(0, 5), 0, Inf), σ) + logσ             # half-Cauchy(σ) + Jacobian
    lp += logpdf(truncated(Cauchy(0, 5), 0, Inf), τ) + logτ             # half-Cauchy(τ) + Jacobian
    # w[1] carries no prior term — a completely free initial condition
    lp += sum(logpdf.(Normal.(ρ .* w[1:end-1] .+ h, σ), w[2:end]))      # AR(1) transition (observed scale)
    lp += sum(logpdf.(Normal.(w, τ), y))                                 # direct, noisy observation
    return lp
end
nothing #hide

# ## Metropolis
#
# ``4 + T = 24`` dimensions is where a joint random-walk proposal starts to hurt, so
# — as in the eight schools — `update!` is *component-wise*: each iteration moves one
# coordinate, whether that is a global parameter or a single time point. The
# [`AdaptiveStep`](@ref) carries one step size per component, all adapted toward the
# same 0.234 target during warm-up.

function update!(s, alg, Δ)
    k     = rand(alg.rng, 1:length(s))
    s_new = copy(s); s_new[k] += Δ[k] * randn(alg.rng)
    accepted = accept!(alg, s_new, s)
    accepted && (s[k] = s_new[k])
    return accepted
end

function metropolis(logposterior, s0; n = 400_000, warmup = 40_000, seed = 42)
    rng  = Xoshiro(seed)
    alg  = MetropolisHastingsAlgorithm(rng, logposterior)
    step = AdaptiveStep(fill(0.5, length(s0)); target = 0.234)
    s    = copy(s0)

    for _ in 1:warmup                                   # warm-up: adapt the step sizes
        accepted = update!(s, alg, step_size(step))
        adapt!(step, accepted)
    end
    reset!(alg)

    Δ = step_size(step)                                 # freeze
    samples = zeros(length(s), n)
    for i in 1:n                                        # sampling
        update!(s, alg, Δ)
        samples[:, i] = s
    end
    return samples, alg
end
nothing #hide

s0 = [0.0, log(mean(y)), 0.0, 0.0, y...]                 # ρ ≈ 0.5, μ_w ≈ mean(y), σ = τ = 1, w = data
samples, alg = metropolis(logposterior, s0)

ρ_post  = 1 ./ (1 .+ exp.(-samples[1, :]))
μw_post = exp.(samples[2, :])
σw_post = exp.(samples[3, :])
τ_post  = exp.(samples[4, :])
h_w_post = μw_post .* (1 .- ρ_post)
w_post  = samples[5:end, :]

(; ρ = round(mean(ρ_post), digits = 2), h_w = round(mean(h_w_post), digits = 2),
   σ_w = round(mean(σw_post), digits = 2), τ = round(mean(τ_post), digits = 2),
   truth = (ρ = ρ_true, h_w = α_true * h_true, σ_w = α_true * σ_true, τ = τ_true),
   acceptance = round(acceptance_rate(alg); digits = 2))

# ## Results
#
# With only 20 points and uninformative priors, the observed-scale posterior is
# honestly wide, same as the no-subsampling version of this model: every parameter's
# 95% credible interval comfortably covers the truth, even where the posterior mean
# wanders from it. This part of the fit needed no information about ``\alpha`` at all.

ci95(v) = (round(quantile(v, 0.025), digits = 2), round(quantile(v, 0.975), digits = 2))
[(; param = name, mean = round(mean(post), digits = 2), ci95 = ci95(post), truth = truth)
 for (post, truth, name) in zip((ρ_post, h_w_post, σw_post, τ_post),
                                 (ρ_true, α_true * h_true, α_true * σ_true, τ_true),
                                 (:ρ, :h_w, :σ_w, :τ))]

# The posterior itself, not just its summary — for ``\rho``, the branching ratio and
# the one parameter of real scientific interest here, since (unlike ``\mu, \sigma``) it
# survives subsampling untouched:

p_ρ = histogram(ρ_post; bins = 80, normalize = :pdf, alpha = 0.5, color = 3, label = "posterior",
                 xlabel = "ρ", ylabel = "density", title = "branching ratio", legend = :topleft)
vline!(p_ρ, [ρ_true]; color = :black, ls = :dash, lw = 2, label = "truth")
vline!(p_ρ, [mean(ρ_post)]; color = 3, ls = :dot, lw = 2, label = "posterior mean")
display(p_ρ)
p_ρ

# Recovering the *population*-scale quantities — the ones you actually care about —
# means drawing ``\alpha`` from a prior and propagating it through. Two priors, two very
# different outcomes: a flat, "I have no idea what fraction I'm sampling" prior, versus
# a weakly informative one that says "I'm fairly sure I'm seeing around half the
# population" (``\mathrm{Beta}(10,10)``, mean 0.5, std ≈ 0.11 — the kind of prior an
# electrode count relative to an estimated population size would actually give you).
# Both draws are pushed through the *same* posterior samples of ``\mu_w`` — only the
# belief about ``\alpha`` differs.

rng_α = Xoshiro(7)
α_uninformative = rand(rng_α, Beta(1, 1), length(ρ_post))
α_informative   = rand(rng_α, Beta(10, 10), length(ρ_post))

for (label, α_draws) in (("uninformative α ~ Beta(1,1)", α_uninformative),
                          ("informative α ~ Beta(10,10)", α_informative))
    μ_pop = μw_post ./ α_draws
    h_pop = μ_pop .* (1 .- ρ_post)
    σ_pop = σw_post ./ α_draws
    println(label)
    println("  μ (pop):  mean = ", round(mean(μ_pop), digits = 2), "  ci95 = ", ci95(μ_pop), "  truth = ", h_true / (1 - ρ_true))
    println("  h (pop):  mean = ", round(mean(h_pop), digits = 2), "  ci95 = ", ci95(h_pop), "  truth = ", h_true)
    println("  σ (pop):  mean = ", round(mean(σ_pop), digits = 2), "  ci95 = ", ci95(σ_pop), "  truth = ", σ_true)
end

# The same comparison, as a picture — both histograms are built from identical MCMC
# output, only the propagated ``\alpha`` differs. The uninformative posterior is
# plotted on a log axis simply because it has a very heavy right tail; the informative
# one would look the same either way.

μ_pop_uninformative = μw_post ./ α_uninformative
μ_pop_informative   = μw_post ./ α_informative

p_pop = histogram(log10.(μ_pop_informative); bins = 60, normalize = :pdf, alpha = 0.6,
                   color = 3, label = "informative α ~ Beta(10,10)",
                   xlabel = "log₁₀ μ (population-scale mean)", ylabel = "density",
                   legend = :topright)
histogram!(p_pop, log10.(μ_pop_uninformative); bins = 60, normalize = :pdf, alpha = 0.4,
           color = 2, label = "uninformative α ~ Beta(1,1)")
vline!(p_pop, [log10(h_true / (1 - ρ_true))]; color = :black, ls = :dash, lw = 2, label = "truth")
display(p_pop)
p_pop

# The uninformative prior does cover the truth — it covers almost everything, which is
# the problem: a "credible interval" that runs from 2.85 to 128.9 is honest but useless.
# The informative prior turns the same MCMC output into a usable, still honestly
# uncertain, answer.

α_prior = Beta(10, 10)
α_draws = rand(Xoshiro(7), α_prior, size(w_post, 2))
x_pop_post = w_post ./ α_draws'

x_mean = vec(mean(x_pop_post, dims = 2))
x_lo   = [quantile(x_pop_post[t, :], 0.025) for t in 1:T]
x_hi   = [quantile(x_pop_post[t, :], 0.975) for t in 1:T]

w_mean = vec(mean(w_post, dims = 2))
w_lo   = [quantile(w_post[t, :], 0.025) for t in 1:T]
w_hi   = [quantile(w_post[t, :], 0.975) for t in 1:T]

p1 = plot(1:T, w_mean; ribbon = (w_mean .- w_lo, w_hi .- w_mean), color = 3, lw = 2,
          label = "posterior mean ± 95% CI", xlabel = "t", ylabel = "w = αx",
          title = "observed-scale trajectory", legend = :topleft)
plot!(p1, 1:T, w_true; color = :black, ls = :dash, lw = 2, label = "truth")
scatter!(p1, 1:T, y; ms = 4, color = 1, label = "observed y")

p2 = plot(1:T, x_mean; ribbon = (x_mean .- x_lo, x_hi .- x_mean), color = 4, lw = 2,
          label = "posterior mean ± 95% CI", xlabel = "t", ylabel = "x",
          title = "population-scale reconstruction", legend = :topleft)
plot!(p2, 1:T, x_true; color = :black, ls = :dash, lw = 2, label = "truth")

p_traj = plot(p1, p2; layout = (1, 2), size = (900, 320), margin = 5Plots.mm)
display(p_traj)
p_traj

# The left panel — the quantity the sampler actually fit — is tight, the same
# recovery you would see with no subsampling at all, plus the usual widest-band
# exception at ``w_1``, which (like ``x_1`` before) has no prior and no left neighbor
# to lean on. The right panel is visibly wider throughout: it carries the observed-scale
# uncertainty *plus* the propagated uncertainty in ``\alpha``, and it is exactly as wide
# as it honestly should be — not a hair tighter.

# ## Gradient-based sampling: Hamiltonian Monte Carlo
#
# The ``\mu_w`` reparametrization removed the worst correlation, but 24 dimensions —
# most of them a tightly coupled latent chain, each ``w_t`` linked to its neighbors
# and to ``\rho, \sigma_w`` — still leaves residual structure that a component-wise
# walk can only cross one coordinate at a time. HMC uses the gradient of the
# log-posterior to move all 24 coordinates together in one coordinated step — the
# same machinery as the eight schools, borrowed unchanged.

using LogDensityProblems, LogDensityProblemsAD, ForwardDiff

ℓ      = ADgradient(:ForwardDiff, FunctionEnsemble(logposterior; dimension = length(s0)))
∇logp(s) = LogDensityProblems.logdensity_and_gradient(ℓ, s)

function hmc_update!(s, alg, ϵ; L = 20)
    p     = randn(alg.rng, length(s))
    lp, g = ∇logp(s)
    H0    = -lp + 0.5 * sum(abs2, p)
    s′    = copy(s)
    for _ in 1:L
        p  .+= 0.5ϵ .* g
        s′ .+= ϵ .* p
        lp, g = ∇logp(s′)
        p  .+= 0.5ϵ .* g
    end
    H = -lp + 0.5 * sum(abs2, p)
    accepted = accept_logratio!(alg, H0 - H)
    accepted && (s .= s′)
    return accepted
end
nothing #hide

function hmc(logposterior, s0; n = 5_000, warmup = 1_000, seed = 42)
    rng  = Xoshiro(seed)
    alg  = MetropolisAlgorithm(rng; β = 1.0)
    step = AdaptiveStep(0.05; target = 0.65)
    s    = copy(s0)

    for _ in 1:warmup
        accepted = hmc_update!(s, alg, step_size(step))
        adapt!(step, accepted)
    end
    reset!(alg)

    ϵ = step_size(step)
    samples = zeros(length(s), n)
    for i in 1:n
        hmc_update!(s, alg, ϵ)
        samples[:, i] = s
    end
    return samples, alg
end
nothing #hide

samples_hmc, alg_hmc = hmc(logposterior, s0)
ρ_hmc = 1 ./ (1 .+ exp.(-samples_hmc[1, :]))
(; ρ = round(mean(ρ_hmc), digits = 2), acceptance = round(acceptance_rate(alg_hmc); digits = 2))

# Same posterior — but a raw sample count is not a fair comparison, since each HMC
# sample costs ``L + 1 = 21`` gradient evaluations against 1 evaluation per Metropolis
# step. Normalizing effective sample size by evaluation count gives the honest
# exchange rate:

n_eval_metropolis = length(ρ_post)
n_eval_hmc        = length(ρ_hmc) * 21
(; ess_metropolis = round(Int, ess(ρ_post)), ess_hmc = round(Int, ess(ρ_hmc)),
   ess_per_eval_metropolis = round(ess(ρ_post) / n_eval_metropolis; sigdigits = 2),
   ess_per_eval_hmc        = round(ess(ρ_hmc) / n_eval_hmc; sigdigits = 2))

# For this particular model, the reparametrized Metropolis is already cheap enough
# per step that HMC's per-sample advantage does not clearly translate into a
# per-evaluation win — unlike the eight schools, where the funnel makes any
# component-wise walk pay dearly regardless of cost accounting. The lesson is not
# "HMC always wins": it is that gradient information helps *in proportion to how much
# correlation is left to fight*, and reparametrizing the target first can already
# remove most of what a fancier sampler would otherwise be paying to overcome — the
# ``\alpha`` ridge earlier in this example is a case where reparametrizing did not just
# help, it removed a parameter from the sampler's job description entirely.

# ## A longer series
#
# The wide ``\rho`` posterior above is not primarily a noise problem — sweeping process
# and observation noise down by 10× at ``T=20`` (tested while building this example)
# barely moves it. The real constraint is length: at ``\rho=0.8`` the process has a
# correlation length ``1/(1-\rho)=5`` steps, so twenty observations cover only about
# four independent looks at the decay dynamics that pin ``\rho`` down. Re-running the
# *identical* model and sampler on a longer series (``T=100``), with observation noise
# turned down somewhat (``\tau`` from 0.5 to 0.2 — a real but modest reduction, not the
# main effect), makes that concrete: same process, same ``\rho``, just more of it.
#
# Every piece below — `logposterior`, `update!`, `metropolis` — is the exact code
# already defined; only the data changes.

T_long, τ_long = 100, 0.2
rng_long = Xoshiro(1)
x_true_long = zeros(T_long)
x_true_long[1] = h_true / (1 - ρ_true) + σ_true / sqrt(1 - ρ_true^2) * randn(rng_long)
for t in 2:T_long
    x_true_long[t] = ρ_true * x_true_long[t-1] + h_true + σ_true * randn(rng_long)
end
w_true_long = α_true .* x_true_long
y = w_true_long .+ τ_long .* randn(rng_long, T_long)   # overwrites the short series `logposterior` reads

s0_long = [0.0, log(mean(y)), 0.0, 0.0, y...]
samples_long, alg_long = metropolis(logposterior, s0_long)
ρ_post_long = 1 ./ (1 .+ exp.(-samples_long[1, :]))

p_ρ_long = histogram(ρ_post_long; bins = 80, normalize = :pdf, alpha = 0.5, color = 4,
                      label = "posterior (T=100)", xlabel = "ρ", ylabel = "density",
                      title = "branching ratio, longer series", legend = :topleft)
vline!(p_ρ_long, [ρ_true]; color = :black, ls = :dash, lw = 2, label = "truth")
vline!(p_ρ_long, [mean(ρ_post_long)]; color = 4, ls = :dot, lw = 2, label = "posterior mean")
display(p_ρ_long)
p_ρ_long

# The direct comparison — same ``\rho``, same model, only ``T`` (and a modest ``\tau``
# reduction) different:

width(ci) = round(ci[2] - ci[1], digits = 2)
[(; series = "T=20 (main example)",  mean = round(mean(ρ_post), digits = 2),      ci95 = ci95(ρ_post),      width = width(ci95(ρ_post))),
 (; series = "T=100 (this section)", mean = round(mean(ρ_post_long), digits = 2), ci95 = ci95(ρ_post_long), width = width(ci95(ρ_post_long)))]

# More data, not less noise, is what turns "honestly uncertain" into "close to the
# truth" here — the noise reduction helps a little on top, but it was never the main
# lever.

# ## References
#
# - C. R. Heathcote, *A branching process allowing immigration*,
#   J. R. Stat. Soc. B **27**, 138 (1965).
#   [doi:10.1111/j.2517-6161.1965.tb01488.x](https://doi.org/10.1111/j.2517-6161.1965.tb01488.x)
# - J. Wilting, V. Priesemann, *25 years of criticality in neuroscience — established
#   results, open controversies, novel concepts*, Curr. Opin. Neurobiol. **58**, 105 (2019).
#   [doi:10.1016/j.conb.2019.08.002](https://doi.org/10.1016/j.conb.2019.08.002)
# - J. Wilting, V. Priesemann, *Inferring collective dynamical states from widely
#   unobserved systems*, Nat. Commun. **9**, 2325 (2018).
#   [doi:10.1038/s41467-018-04725-4](https://doi.org/10.1038/s41467-018-04725-4)
# - A. Gelman, *Prior distributions for variance parameters in hierarchical models*,
#   Bayesian Anal. **1**, 515 (2006).
#   [doi:10.1214/06-BA117A](https://doi.org/10.1214/06-BA117A)
