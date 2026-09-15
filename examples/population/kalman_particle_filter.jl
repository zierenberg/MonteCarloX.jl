# # Sequential Filtering — Kalman and Particle Filters
#
# A state drifts randomly and is watched through noise:
#
# ```math
# x_t = x_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, q)
# ```
# ```math
# y_t = x_t + v_t, \qquad v_t \sim \mathcal{N}(0, r)
# ```
#
# the simplest possible linear-Gaussian state-space model (the textbook "local level"
# model). Unlike [AR(1) with immigration](ar1.md), the question here is not "what
# parameters generated this data" — ``q`` and ``r`` are taken as known throughout. The
# question is **filtering**: given ``y_1,\dots,y_t`` so far, what is a good running
# estimate of ``x_t``, updated as each new point arrives, without re-processing the
# whole history every time?
#
# Three answers, in increasing order of sophistication:
#
# 1. A **naive online estimate** — the obvious thing to reach for without thinking
#    about the state-space structure at all.
# 2. **Exact Bayesian filtering** — for a linear-Gaussian model, the posterior
#    ``p(x_t \mid y_{1:t})`` is Gaussian in closed form at every ``t``: the Kalman
#    filter. No Monte Carlo needed.
# 3. A **particle filter** — Monte Carlo filtering that does not need linearity or
#    Gaussian noise. It is validated here against the Kalman filter, its
#    closed-form ground truth, before it is ever the only option.
#
# The particle filter below is the original member of the broader family the package's
# [Population Monte Carlo](../algorithms/population_monte_carlo.md) roadmap describes —
# a population of weighted samples carried through a sequence of distributions, here
# indexed by time rather than by an annealing schedule.

using Random, Distributions, StatsBase, Plots
using MonteCarloX

# ## Synthetic data
#
# ``T = 100`` steps, with the state's own random-walk noise ``q`` an order of magnitude
# smaller than the observation noise ``r`` — individual observations are quite noisy,
# but the state itself does not move far between steps. A diffuse initial prior
# (``P_0 = 10``) starts the filters knowing essentially nothing about ``x_1``.

T  = 100
q  = 0.1
r  = 2.0
m0, P0 = 0.0, 10.0

rng = Xoshiro(1)
x_true = zeros(T)
x_true[1] = m0 + sqrt(P0) * randn(rng)
for t in 2:T
    x_true[t] = x_true[t-1] + sqrt(q) * randn(rng)
end
y = x_true .+ sqrt(r) .* randn(rng, T)
nothing #hide

# ## A naive online estimate
#
# Without the state-space structure in mind, the obvious online estimate is the
# running mean of everything observed so far: ``\hat x_t = \frac{1}{t}\sum_{i=1}^t
# y_i``, updated in ``O(1)`` per step. It implicitly assumes the world is
# **stationary** — every past observation is exactly as informative about ``x_t`` as
# the newest one. That assumption is exactly what fails here: ``x_t`` drifts, so old
# observations become stale, and the running mean drags further and further behind
# the true state as ``t`` grows, with no way to know it is doing so — a point estimate
# and nothing else, no notion of its own uncertainty.

x_naive = cumsum(y) ./ (1:T)
nothing #hide

# ## Exact Bayesian filtering: the Kalman filter
#
# For a linear-Gaussian model, the posterior ``p(x_t \mid y_{1:t})`` stays Gaussian at
# every step, so filtering reduces to propagating a mean and a variance through two
# closed-form steps:
#
# ```math
# \text{predict:} \quad m_{t|t-1} = m_{t-1}, \qquad P_{t|t-1} = P_{t-1} + q
# ```
# ```math
# \text{update:} \quad K_t = \frac{P_{t|t-1}}{P_{t|t-1}+r}, \qquad
# m_t = m_{t|t-1} + K_t(y_t - m_{t|t-1}), \qquad P_t = (1-K_t)P_{t|t-1}
# ```
#
# ``K_t``, the Kalman gain, is the whole story: it is how much the new observation
# should move the estimate, and — unlike the naive estimate's implicit, frozen weight
# on every past point — it is *recomputed every step* from how uncertain the filter
# currently is. Started from the diffuse ``P_0 = 10``, it trusts the first observation
# almost completely (``K_1 \approx 0.83``) and settles within a handful of steps to a
# steady-state value set by the ratio ``q/r`` alone.

function kalman_filter(y, q, r, m0, P0)
    T = length(y)
    m, P = zeros(T), zeros(T)
    m_pred, P_pred = m0, P0
    for t in 1:T
        if t > 1
            m_pred = m[t-1]
            P_pred = P[t-1] + q
        end
        K = P_pred / (P_pred + r)
        m[t] = m_pred + K * (y[t] - m_pred)
        P[t] = (1 - K) * P_pred
    end
    return m, P
end
nothing #hide

m, P = kalman_filter(y, q, r, m0, P0)
(; rmse_naive = round(sqrt(mean((x_naive .- x_true) .^ 2)), digits = 3),
   rmse_kalman = round(sqrt(mean((m .- x_true) .^ 2)), digits = 3))

# The gap is the cost of pretending the process is stationary when it is not.

# ## Particle filter
#
# The Kalman filter is only exact because the model is linear-Gaussian. A **particle
# filter** solves the same sequential problem — carry ``p(x_t \mid y_{1:t})`` forward
# one step at a time — by representing it as a weighted population of samples instead
# of a closed-form mean and variance, which is what lets it survive nonlinear
# transitions or non-Gaussian noise where the Kalman recursion simply does not apply.
# Here it is deliberately run on the model that *does* have a closed form, so its
# output has an exact answer to be checked against.
#
# The bootstrap filter [Gordon, Salmond & Smith 1993] propagates each particle through
# the transition, then reweights it by how well it explains the new observation:
# ``w_t^{(i)} \propto w_{t-1}^{(i)}\, p(y_t \mid x_t^{(i)})``. That "``\propto``" hides
# the one place this is easy to get subtly wrong: the weight update is
# **multiplicative across time** — each step's likelihood multiplies (in log space,
# adds to) whatever weight the particle already carried — not a fresh per-step weight
# that discards the history. Reset it every step and the filter still runs, still looks
# plausible, and is quietly wrong in a way more particles never fixes (the tell: its
# distance to the Kalman filter stays flat as ``N`` grows instead of shrinking).
#
# Left unresampled, weights concentrate onto a shrinking handful of particles —
# tracked here with [`ImportanceWeights`](@ref) and its Kish [`ess`](@ref), the same
# reweighting primitives the package uses for statistical-mechanics ensembles. When
# that effective sample size drops below half the population, particles are refreshed
# by systematic resampling in proportion to their weight, and the accumulated weight
# resets — the post-resampling population is, by construction, unweighted again.

function systematic_resample(rng, w::AbstractVector)
    n = length(w)
    positions = (rand(rng) .+ (0:n-1)) ./ n
    idx = zeros(Int, n)
    cw = cumsum(w); cw[end] = 1.0            # guard against floating-point drift
    i = 1
    for (j, p) in enumerate(positions)
        while cw[i] < p
            i += 1
        end
        idx[j] = i
    end
    return idx
end

function particle_filter(rng, y, N; q, r, m0, P0, resample_threshold = 0.5)
    T = length(y)
    x_pf, P_pf, ess_trace = zeros(T), zeros(T), zeros(T)
    particles = m0 .+ sqrt(P0) .* randn(rng, N)
    logw = zeros(N)                                        # accumulates across steps
    for t in 1:T
        if t > 1
            particles .+= sqrt(q) .* randn(rng, N)          # propagate
        end
        logw .+= logpdf.(Normal.(particles, sqrt(r)), y[t]) # reweight (multiplicative)
        iw = ImportanceWeights(logw)
        w  = weights(iw)
        x_pf[t] = mean(particles, w)
        P_pf[t] = var(particles, w; corrected = false)
        ess_trace[t] = ess(iw)
        if ess_trace[t] < resample_threshold * N            # adaptive resampling
            particles = particles[systematic_resample(rng, w)]
            logw = zeros(N)                                 # reset: unweighted again
        end
    end
    return (; x_pf, P_pf, ess_trace)
end
nothing #hide

N  = 1_000
pf = particle_filter(Xoshiro(7), y, N; q, r, m0, P0)
(; rmse_pf_truth  = round(sqrt(mean((pf.x_pf .- x_true) .^ 2)), digits = 3),
   rmse_pf_kalman = round(sqrt(mean((pf.x_pf .- m) .^ 2)), digits = 4))

# 1,000 particles track the true state about as well as the Kalman filter does, and
# track the Kalman filter itself far more closely than either tracks the truth — which
# is exactly the point: the particle filter's job is to reproduce the exact Bayesian
# answer, not to beat it.

# ## Why resampling: the effective sample size

ess_no_resample = let
    T = length(y)
    rng_nr = Xoshiro(7)
    ess_trace = zeros(T)
    particles = m0 .+ sqrt(P0) .* randn(rng_nr, N)
    logw = zeros(N)
    for t in 1:T
        if t > 1
            particles .+= sqrt(q) .* randn(rng_nr, N)
        end
        logw .+= logpdf.(Normal.(particles, sqrt(r)), y[t])
        ess_trace[t] = ess(ImportanceWeights(logw))
    end
    ess_trace
end

plot(1:T, ess_no_resample; color = 2, lw = 2, label = "no resampling",
     xlabel = "t", ylabel = "effective sample size", legend = :topright)
plot!(1:T, pf.ess_trace; color = 3, lw = 2, label = "adaptive resampling (N/2 threshold)")
hline!([N]; color = :black, ls = :dot, label = "N = $N")

# Without resampling, ``\mathrm{ESS}`` decays essentially monotonically as weight mass
# collects on whichever particles happened to land near the truth early on — by
# ``t=100`` a population of 1,000 is worth only a handful of independent samples, no
# matter how large ``N`` was to start with. Adaptive resampling refreshes the
# population whenever ``\mathrm{ESS}`` would drop below ``N/2``, and keeps the filter
# perpetually well-supported instead of decaying once and never recovering.

# ## Comparison

plot(1:T, m; ribbon = 1.96 .* sqrt.(P), color = 3, lw = 2, fillalpha = 0.25,
     label = "Kalman (exact) ± 95%", xlabel = "t", ylabel = "x", legend = :topleft)
plot!(1:T, pf.x_pf; ribbon = 1.96 .* sqrt.(pf.P_pf), color = 4, lw = 1.5, ls = :dash,
      fillalpha = 0.15, label = "particle filter (N=$N) ± 95%")
plot!(1:T, x_naive; color = 2, lw = 2, label = "naive running mean")
plot!(1:T, x_true; color = :black, ls = :dot, lw = 2, label = "truth")
scatter!(1:T, y; ms = 3, color = :gray, alpha = 0.5, label = "observed y")

# The naive estimate — no state-space structure, no adaptive gain — increasingly lags
# the drifting truth as the series goes on. The Kalman filter and the particle filter,
# solving the same Bayesian filtering problem by different means, are close enough to
# be difficult to tell apart on this plot; the ribbon is genuine posterior uncertainty
# in both cases, not a decoration — something the naive estimate simply does not have.

# ## Validation: the particle filter is not a different method, just a slower one
#
# For a linear-Gaussian model the bootstrap filter has nothing to offer over the exact
# recursion — its entire value is that it keeps working once linearity or Gaussianity
# is gone, which the Kalman filter cannot say. What can be checked here is whether it
# is a *correct* approximation: its distance to the Kalman filter should shrink as
# ``N`` grows, at the standard Monte Carlo rate ``O(1/\sqrt{N})``.

[(; N = Ntest,
   rmse_to_kalman = round(sqrt(mean((particle_filter(Xoshiro(7), y, Ntest; q, r, m0, P0).x_pf .- m) .^ 2)), digits = 4))
 for Ntest in (20, 100, 1_000, 10_000)]

# Tenfold increases in ``N`` shrink the gap to the exact answer by close to the
# ``\sqrt{10} \approx 3.16`` a consistent Monte Carlo estimator predicts — the
# particle filter is not a qualitatively different answer from the Kalman filter, just
# a Monte Carlo approximation to the same posterior that happens, on this model, to
# have a cheaper exact alternative.

# ## References
#
# - R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems*,
#   J. Basic Eng. **82**, 35 (1960).
#   [doi:10.1115/1.3662552](https://doi.org/10.1115/1.3662552)
# - N. J. Gordon, D. J. Salmond, A. F. M. Smith, *Novel approach to nonlinear/non-Gaussian
#   Bayesian state estimation*, IEE Proc. F **140**, 107 (1993).
#   [doi:10.1049/ip-f-2.1993.0015](https://doi.org/10.1049/ip-f-2.1993.0015)
# - A. Doucet, A. M. Johansen, *A Tutorial on Particle Filtering and Smoothing: Fifteen
#   years later*, in *The Oxford Handbook of Nonlinear Filtering*, Oxford University
#   Press (2011), pp. 656–705.
