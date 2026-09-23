# # Simulation-Based Inference — Approximate Bayesian Computation
#
# Run with `julia examples/inference/sbi.jl` from anywhere in the repo — no
# `--project` flag needed, the line below activates `examples/` itself.

import Pkg; Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()  #src
#
# Simulation-based inference (SBI, a.k.a. likelihood-free inference) applies when
# you can *simulate* data but cannot write down the likelihood ``p(x\mid\theta)``.
# **Approximate Bayesian computation (ABC)** is the classical approach: draw
# ``\theta`` from the prior, simulate ``x``, and keep (or move toward) draws whose
# `x` resembles the observation `y`.
#
# | method | idea | sampler used here |
# |:--|:--|:--|
# | ABC rejection | keep prior draws whose simulated `x` is close to `y` | plain filter |
# | ABC-MCMC | Markov chain that only moves to a `θ'` whose simulation is close to `y` | [`accept_logratio!`](@ref) |
# | NPE | train ``q_\phi(\theta\mid s)``, evaluate once at `s_obs` | none — amortized |
# | NLE | train ``q_\phi(s\mid\theta)``, use it as a likelihood | [`accept!`](@ref) |
#
# **NPE**, **NLE**, **NRE** (Neural Posterior/Likelihood/Ratio Estimation) replace
# "close to `y`" with a trained model. The last two sections work through NPE and
# NLE on the same simulations — NLE is the one that hands MonteCarloX a chain to run.
#
# ## Simulator: a stationary Ornstein–Uhlenbeck process
#
# [OU](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process):
# ``dx = -\kappa(x-\mu)\,dt + \sigma\,dW``, ``\sigma=\sqrt{2D}``, started from its own
# stationary distribution. We infer ``\theta=(\kappa,D)`` with ``\mu=0`` fixed. Its
# transition is Gaussian and known exactly, so we simulate with it directly —
# `StochasticDiffEq.jl` is the right tool for an SDE *without* a closed form.

using Random, Distributions, StatsBase, Plots

CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"   #hide
shrink(full, small) = CI_MODE ? small : full                             #hide
nothing #hide

n_obs, dt = 15, 0.4

function simulate_ou(θ, rng)
    κ, D = θ
    a, v = exp(-κ * dt), D / κ * (1 - exp(-2κ * dt))
    x    = zeros(n_obs + 1)
    x[1] = sqrt(D / κ) * randn(rng)              # stationary initial condition
    for i in 2:n_obs+1
        x[i] = a * x[i-1] + sqrt(v) * randn(rng)
    end
    return x
end

function loglik_exact(θ, x)
    κ, D = θ
    a, v = exp(-κ * dt), D / κ * (1 - exp(-2κ * dt))
    ll = logpdf(Normal(0, sqrt(D / κ)), x[1])
    for i in 2:length(x)
        ll += logpdf(Normal(a * x[i-1], sqrt(v)), x[i])
    end
    return ll
end
nothing #hide

# ## Observed data, prior, and the exact posterior
#
# Two parameters, so the reference posterior comes from evaluating the exact density
# on a grid and normalizing — as in `gaussian.jl`, no chain needed.

truth = [1.2, 0.6]
y_obs = simulate_ou(truth, Xoshiro(2))

logprior(θ) = all(θ .> 0) ?
    logpdf(LogNormal(log(1.0), 0.5), θ[1]) + logpdf(LogNormal(log(0.5), 0.5), θ[2]) : -Inf
rand_prior(rng) = [rand(rng, LogNormal(log(1.0), 0.5)), rand(rng, LogNormal(log(0.5), 0.5))]
logposterior_exact(θ) = all(θ .> 0) ? logprior(θ) + loglik_exact(θ, y_obs) : -Inf

κg, Dg = range(0.2, 3.0; length = 300), range(0.05, 2.0; length = 300)
κg, Dg = range(0.2, 3.0; length = shrink(300, 80)), range(0.05, 2.0; length = shrink(300, 80))  #hide
logp   = [logposterior_exact([κ, D]) for D in Dg, κ in κg]
w      = exp.(logp .- maximum(logp))
κ_grid = sum(w .* κg') / sum(w)
D_grid = sum(w .* Dg) / sum(w)
(; κ_grid, D_grid)

# ## Summary statistics
#
# Read them off `loglik_exact`: expanding ``\sum_i (x_i - a x_{i-1})^2`` shows the exact
# log-likelihood depends on the data *only* through the two second moments
#
# ```math
# m_0 = \frac{1}{n}\sum_i x_i^2 \approx \frac{D}{\kappa}, \qquad
# m_1 = \frac{1}{n-1}\sum_i x_i x_{i+1} \approx \frac{D}{\kappa}\,e^{-\kappa\,dt},
# ```
#
# plus two boundary terms from the stationary start. Keeping ``(m_0, m_1)`` is therefore
# sufficient up to those boundary terms, and their ratio ``m_1/m_0`` pins down ``\kappa``
# while ``m_0`` pins down ``D``.
#
# !!! warning "Do not reach for `var` and `cor` here"
#     The obvious choices — sample variance and lag-1 autocorrelation — subtract the
#     *sample* mean, silently discarding the modelling assumption that ``\mu = 0`` is
#     known. That is not a rounding error: with `var`/`cor` every sampler below lands
#     ``\kappa`` low by ≈ 0.1–0.3 across observations, a shift that does **not** shrink
#     as ``\varepsilon \to 0`` because it is not a tolerance effect. You are always
#     sampling ``p(\theta \mid s_{\text{obs}})``, never ``p(\theta \mid y_{\text{obs}})``;
#     the two differ by exactly what the statistics throw away. All four samplers here
#     would have agreed with each other while all four were off — so agreement between
#     likelihood-free methods checks the *samplers*, not the *inference*.

summary_stats(x) = [mean(abs2, x), sum(x[1:end-1] .* x[2:end]) / (length(x) - 1)]
s_obs = summary_stats(y_obs)
(; s_obs, m0_predicted = truth[2] / truth[1],
   m1_predicted = truth[2] / truth[1] * exp(-truth[1] * dt))

# ## Training pool
#
# `(\theta, s)` pairs from the prior predictive, drawn once and reused below.

function simulate_prior_predictive(rng, n)
    θs = [rand_prior(rng) for _ in 1:n]
    ss = [summary_stats(simulate_ou(θ, rng)) for θ in θs]
    return θs, ss
end

n_train = 20_000
n_train = shrink(n_train, 4_000)   #hide
θs_train, ss_train = simulate_prior_predictive(Xoshiro(3), n_train)
θmat = permutedims(reduce(hcat, θs_train))     # n×2
smat = permutedims(reduce(hcat, ss_train))     # n×2
s_std = vec(std(smat, dims = 1))

distance(s, s′) = sqrt(sum(((s .- s′) ./ s_std) .^ 2))
nothing #hide

# ## ABC rejection

ρs = [distance(s, s_obs) for s in eachrow(smat)]
ε  = quantile(ρs, 0.01)
θ_abc_rej = θmat[ρs .≤ ε, :]
θ0_seed   = vec(mean(θ_abc_rej, dims = 1))      # the *mean* of the accepted cloud, not one
                                                 # arbitrary draw from it — a single accepted
                                                 # point can be an outlier and strand ABC-MCMC
                                                 # far from the posterior, given how slowly it mixes
(; ε, n_accepted = size(θ_abc_rej, 1), θ0_seed = round.(θ0_seed; digits = 2))

# ## ABC-MCMC
#
# Marjoram et al. [2003]: propose ``\theta'``, simulate, and accept with the
# Metropolis rule but with the likelihood ratio replaced by the indicator kernel
# ``\mathbb{1}[\rho(s',s_{\text{obs}})\le\varepsilon]``. Since the current state
# always satisfies the kernel, acceptance reduces to `min(1, prior ratio)` whenever
# the proposal lands inside ``\varepsilon`` — handed to [`accept_logratio!`](@ref),
# MCX's raw log-acceptance-ratio primitive (ensemble slot unused: see closing
# notes). The proposal uses `rand`, not `randn` — only symmetry matters here, not
# shape.
#
# !!! warning "Do not adapt the step size here"
#     ABC-MCMC acceptance is *capped* by the probability that a fresh simulation at
#     the current ``\theta`` lands inside ``\varepsilon`` — a few percent, and it does
#     **not** go to 1 as ``\Delta \to 0``, because every proposal re-simulates. Feeding
#     this into an [`AdaptiveStep`](@ref) with a usual target (0.234, 0.3) asks for an
#     acceptance the chain can never reach, so ``\Delta`` collapses geometrically and
#     the chain freezes in place. We instead fix the proposal width from the width of
#     the ABC-rejection cloud, which already estimates the posterior scale.

function update_abc!(θ, alg, Δ, ε)
    θ′ = θ .+ Δ .* (2 .* rand(alg.rng, 2) .- 1)
    if !all(θ′ .> 0) || distance(summary_stats(simulate_ou(θ′, alg.rng)), s_obs) > ε
        accept_logratio!(alg, -Inf)
        return false
    end
    accepted = accept_logratio!(alg, logprior(θ′) - logprior(θ))
    accepted && (θ .= θ′)
    return accepted
end

using MonteCarloX

function abc_mcmc(θ0, ε, Δ; n = 50_000, warmup = 5_000, seed = 10)
    rng = Xoshiro(seed)
    alg = MetropolisHastingsAlgorithm(rng, θ -> 0.0)    # ensemble unused: see closing notes
    θ   = copy(θ0)

    for _ in 1:warmup
        update_abc!(θ, alg, Δ, ε)
    end
    reset!(alg)

    samples = zeros(2, n)
    for i in 1:n
        update_abc!(θ, alg, Δ, ε)
        samples[:, i] = θ
    end
    return samples, alg
end

ε_mcmc = quantile(ρs, 0.10)                     # looser than rejection: a chain must keep re-simulating
Δ_mcmc = vec(std(θ_abc_rej, dims = 1))          # proposal width ≈ posterior width
n_abc, warmup_abc = 50_000, 5_000
n_abc, warmup_abc = shrink(n_abc, 4_000), shrink(warmup_abc, 500)   #hide
samples_abc, alg_abc = abc_mcmc(θ0_seed, ε_mcmc, Δ_mcmc; n = n_abc, warmup = warmup_abc)
(; acceptance = round(acceptance_rate(alg_abc); digits = 3),
   mean = round.(vec(mean(samples_abc, dims = 2)); digits = 2),
   std  = round.(vec(std(samples_abc, dims = 2)); digits = 2))

# ## Comparison against the exact posterior
#
# Both ABC variants cover the exact posterior and agree with each other — which is the
# check that matters, since they share the same ``\varepsilon``-kernel. They also agree
# with the *exact* posterior, and that is not automatic: it is what the sufficient
# statistics bought. Throwing away 14 of the 16 data points costs nothing here only
# because those two numbers carry (nearly) everything the likelihood ever sees.

κ_marg = vec(sum(w, dims = 1)) ./ (sum(w) * step(κg))
nothing #hide

function plot_joint(title)
    p = contour(κg, Dg, w; levels = 8, color = :black, colorbar = false, xlabel = "κ",
                ylabel = "D", title = title, xlims = extrema(κg), ylims = extrema(Dg))
    plot!(p, Float64[], Float64[]; lw = 2, color = :black, label = "exact")   # legend proxy
end

mark_truth!(p) = scatter!(p, [truth[1]], [truth[2]]; ms = 7, color = :red,
                          marker = :star5, label = "truth")

function plot_marginal(title)
    plot(κg, κ_marg; lw = 3, color = :black, label = "exact", xlabel = "κ",
         ylabel = "density", title = title, xlims = extrema(κg))
end
nothing #hide

pjoint = plot_joint("posterior over (κ, D)")
scatter!(pjoint, θ_abc_rej[:, 1], θ_abc_rej[:, 2]; ms = 2, alpha = 0.4, color = 1, label = "ABC rejection")
scatter!(pjoint, samples_abc[1, 1:25:end], samples_abc[2, 1:25:end]; ms = 1.5, alpha = 0.2, color = 2, label = "ABC-MCMC")
mark_truth!(pjoint)

pκ = plot_marginal("marginal of κ")
stephist!(pκ, θ_abc_rej[:, 1]; normalize = :pdf, lw = 2, color = 1, label = "ABC rej.")
stephist!(pκ, samples_abc[1, :]; normalize = :pdf, lw = 2, color = 2, label = "ABC-MCMC")
vline!(pκ, [truth[1]]; color = :red, ls = :dash, label = "")

plot(pjoint, pκ; layout = (1, 2), size = (950, 360), margin = 4Plots.mm)

# ## Conditional neural density estimators
#
# ABC's accept/reject rule is a crude stand-in for a *conditional density
# estimator*. The neural family [Cranmer, Brehmer, Louppe 2020] fits one on the
# same `(θ, s)` pairs from `simulate_prior_predictive` — no `ε`, no distance:
#
# - **NPE** fits ``q_\phi(\theta\mid s)`` directly — one forward pass, no MCMC.
# - **NLE** fits ``q_\phi(s\mid\theta)``, a learned `loglik_exact`, which then goes
#   into an ordinary Metropolis chain.
# - **NRE** trains a classifier on real vs. shuffled `(θ,s)` pairs; its logit
#   approximates the likelihood-to-evidence ratio, again for MCMC.
#
# So "neural" and "MCMC" are not alternatives: only NPE avoids a chain, by learning
# the posterior itself. NLE and NRE learn the *integrand* and still need a sampler —
# and that sampler is plain MonteCarloX, exactly as in [gaussian.jl](gaussian.md).
# Both are worked below from one shared density network.
#
# Reference software: the `sbi` Python toolkit [Tejero-Cantero et al. 2020], from
# Jakob Macke's group, who apply this to neuroscience models directly
# ([Gao, Deistler, Macke 2024]; Boelts et al. 2022).
#
# ## A Gaussian density network with `Flux.jl`
#
# The smallest honest estimator maps one 2-vector to the mean and covariance of a
# Gaussian over another — a one-component mixture density network, trained by maximum
# likelihood. `Flux.jl` supplies the layers, `Adam`, minibatches, and gradients; no
# hand-written backpropagation. The covariance is parametrized by a Cholesky factor
# so it stays positive-definite by construction, and its negative log-likelihood is
# the loss. The *same* `density_net` serves NPE (`s → θ`) and NLE (`θ → s`).

using Flux

function gaussian_nll(model, X, Y)
    out = model(X)                                        # 5×n: μ₁, μ₂, Cholesky factor
    l11, l21, l22 = softplus.(out[3, :]) .+ 1f-3, out[4, :], softplus.(out[5, :]) .+ 1f-3
    z1 = (Y[1, :] .- out[1, :]) ./ l11
    z2 = (Y[2, :] .- out[2, :] .- l21 .* z1) ./ l22
    return mean(0.5f0 .* (z1 .^ 2 .+ z2 .^ 2) .+ log.(l11) .+ log.(l22))
end

function density_net(X, Y; epochs = 40, seed = 7)
    Random.seed!(seed)
    model     = Chain(Dense(2, 32, relu), Dense(32, 32, relu), Dense(32, 5))
    opt_state = Flux.setup(Adam(1e-3), model)
    loader    = Flux.DataLoader((X, Y); batchsize = 256, shuffle = true)
    for _ in 1:epochs, (xb, yb) in loader
        _, grads = Flux.withgradient(m -> gaussian_nll(m, xb, yb), model)
        Flux.update!(opt_state, model, grads[1])
    end
    return model
end

gauss(model, x) = (out = model(reshape(x, 2, 1));                    # (μ, Cholesky factor L)
                   (Float64.(out[1:2, 1]),
                    [Float64(softplus(out[3, 1]))+1e-3  0.0
                     Float64(out[4, 1])                 Float64(softplus(out[5, 1]))+1e-3]))

logpdf_gauss(μ, L, y) = (z1 = (y[1] - μ[1]) / L[1, 1];
                         z2 = (y[2] - μ[2] - L[2, 1] * z1) / L[2, 2];
                         -0.5 * (z1^2 + z2^2) - log(L[1, 1]) - log(L[2, 2]) - log(2π))
nothing #hide

# ### Coordinates
#
# A Gaussian on ``(\kappa, D)`` would put mass on negative rates, so the network works
# in ``\log\theta`` — unconstrained, and matching the LogNormal prior. Likewise ``m_0``
# is positive and heavy-tailed, so it enters as ``\log m_0``; ``m_1`` can be negative and
# stays as it is. Both sides are then standardized, which is what makes a small net
# train in seconds.

logθmat = log.(θmat)
zmat    = hcat(log.(smat[:, 1]), smat[:, 2])                 # (log m₀, m₁)
z_obs   = [log(s_obs[1]), s_obs[2]]

θ_mean, θ_std = vec(mean(logθmat, dims = 1)), vec(std(logθmat, dims = 1))
z_mean, z_std = vec(mean(zmat,    dims = 1)), vec(std(zmat,    dims = 1))
Θ = Float32.(permutedims((logθmat .- θ_mean') ./ θ_std'))    # 2×n standardized log-parameters
Z = Float32.(permutedims((zmat    .- z_mean') ./ z_std'))    # 2×n standardized statistics
z_obs_n = Float32.((z_obs .- z_mean) ./ z_std)
nothing #hide

# ## NPE: the posterior in one forward pass

n_epochs, n_draw, warmup_nle = 40, 20_000, 2_000
n_epochs, n_draw, warmup_nle = shrink(n_epochs, 4), shrink(n_draw, 4_000), shrink(warmup_nle, 400)  #hide

npe_model = density_net(Z, Θ; epochs = n_epochs)
μ_npe, L_npe = gauss(npe_model, z_obs_n)
samples_npe  = exp.((μ_npe .* θ_std .+ θ_mean) .+ (θ_std .* L_npe) * randn(Xoshiro(50), 2, n_draw))
(; mean_npe = round.(vec(mean(samples_npe, dims = 2)); digits = 2),
   std_npe  = round.(vec(std(samples_npe,  dims = 2)); digits = 2))

# `s_obs` in, a posterior out — no simulations at inference time, no `ε`, no chain.
# The same trained net answers any other observation instantly: that is *amortization*,
# and it is the one thing ABC can never do.
#
# ## NLE: a learned likelihood, sampled with MonteCarloX
#
# Swap the arguments and the identical network learns ``q_\phi(s\mid\theta)`` instead.
# Evaluated at the fixed `z_obs` it is a function of ``\theta`` alone — a drop-in
# replacement for `loglik_exact`, and from there everything is the ordinary
# [`MetropolisHastingsAlgorithm`](@ref) of the other inference examples: a
# [`FunctionEnsemble`](@ref) over `logprior + learned loglik`, [`accept!`](@ref) on the
# proposed pair, and [`AdaptiveStep`](@ref) — which *is* safe here, because this
# acceptance is a deterministic density ratio with no simulation noise capping it.

function loglik_nle(θ)
    all(θ .> 0) || return -Inf
    μ, L = gauss(nle_model, Float32.((log.(θ) .- θ_mean) ./ θ_std))
    return logpdf_gauss(μ, L, (z_obs .- z_mean) ./ z_std)
end
logposterior_nle(θ) = all(θ .> 0) ? logprior(θ) + loglik_nle(θ) : -Inf

function update!(θ, alg, Δ)
    θ′       = θ .+ Δ .* randn(alg.rng, 2)
    accepted = accept!(alg, θ′, θ)
    accepted && (θ .= θ′)
    return accepted
end

function metropolis(logposterior; n = 20_000, warmup = 2_000, Δ0 = [0.3, 0.15], seed = 11)
    rng  = Xoshiro(seed)
    alg  = MetropolisHastingsAlgorithm(rng, logposterior)
    step = AdaptiveStep(Δ0; target = 0.3)
    θ    = [1.0, 0.5]

    for _ in 1:warmup
        adapt!(step, update!(θ, alg, step_size(step)))
    end
    reset!(alg)

    Δ, samples = step_size(step), zeros(2, n)
    for i in 1:n
        update!(θ, alg, Δ)
        samples[:, i] = θ
    end
    return samples, alg
end

nle_model = density_net(Θ, Z; epochs = n_epochs)
samples_nle, alg_nle = metropolis(logposterior_nle; n = n_draw, warmup = warmup_nle)
(; acceptance = round(acceptance_rate(alg_nle); digits = 2),
   mean_nle = round.(vec(mean(samples_nle, dims = 2)); digits = 2),
   std_nle  = round.(vec(std(samples_nle,  dims = 2)); digits = 2))

# ## The neural estimators against the exact posterior
#
# NPE and NLE were trained on the *same* 20 000 simulations that ABC filtered, and land
# on the same answer as each other, as ABC, and as the exact posterior. Accuracy is not
# what separates these methods — they all target ``p(\theta\mid s_{\text{obs}})``, so they
# stand or fall together on the statistics. The cost is what differs: ABC threw away
# 99 % of its simulations, both networks used all of them, and NPE needs no simulator
# call at all once trained — a new observation is one forward pass. That amortization is
# the thing ABC can never offer.

pjoint2 = plot_joint("neural estimators vs. exact")
scatter!(pjoint2, samples_npe[1, 1:10:end], samples_npe[2, 1:10:end]; ms = 1.5, alpha = 0.2, color = 6, label = "NPE")
scatter!(pjoint2, samples_nle[1, 1:10:end], samples_nle[2, 1:10:end]; ms = 1.5, alpha = 0.2, color = 4, label = "NLE + Metropolis")
mark_truth!(pjoint2)

pκ2 = plot_marginal("marginal of κ")
stephist!(pκ2, samples_npe[1, :]; normalize = :pdf, lw = 2, color = 6, label = "NPE")
stephist!(pκ2, samples_nle[1, :]; normalize = :pdf, lw = 2, color = 4, label = "NLE")
stephist!(pκ2, samples_abc[1, :]; normalize = :pdf, lw = 2, color = 2, ls = :dot, label = "ABC-MCMC")
vline!(pκ2, [truth[1]]; color = :red, ls = :dash, label = "")

plot(pjoint2, pκ2; layout = (1, 2), size = (950, 360), margin = 4Plots.mm)
#
# ## What's missing
#
# Neither sampler needed new API: [`accept_logratio!`](@ref) is exactly the right shape
# for an acceptance rule that isn't a `logweight` difference, and NLE is a plain
# [`MetropolisHastingsAlgorithm`](@ref) over a learned log-density. Three smaller gaps:
#
# - **No prior-predictive helper.** `simulate_prior_predictive` (draw `θ`, simulate,
#   reduce to statistics) is boilerplate every SBI method needs, not just ABC.
# - **`accept_logratio!` needs a placeholder ensemble** (`θ -> 0.0` above), since the
#   whole acceptance decision bypasses `logweight` — harmless, but only because we
#   never call `ensemble(alg)`.
# - **[`AdaptiveStep`](@ref) is unsafe for simulation-based acceptance.** Its target is
#   unreachable whenever acceptance is capped below 1 by a stochastic kernel, and it
#   then shrinks the step to zero without complaint. A guard — stop adapting, or warn,
#   when the step collapses while the acceptance stays pinned — would have caught the
#   frozen chain immediately.
#
# NPE alone uses none of MonteCarloX, and should not: amortized inference has no chain
# to run. Small, general, and worth adding if more SBI examples follow.
#
# ## References
#
# - [Wikipedia: Approximate Bayesian computation](https://en.wikipedia.org/wiki/Approximate_Bayesian_computation)
# - [Wikipedia: Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
# - P. Marjoram, J. Molitor, V. Plagnol, S. Tavaré, *Markov chain Monte Carlo without likelihoods*, PNAS **100**, 15324 (2003). [doi:10.1073/pnas.0306899100](https://doi.org/10.1073/pnas.0306899100)
# - K. Cranmer, J. Brehmer, G. Louppe, *The frontier of simulation-based inference*, PNAS **117**, 30055 (2020) — covers NPE/NLE/NRE. [doi:10.1073/pnas.1912789117](https://doi.org/10.1073/pnas.1912789117)
# - A. Tejero-Cantero et al., *sbi: A toolkit for simulation-based inference*, J. Open Source Softw. **5**, 2505 (2020). [doi:10.21105/joss.02505](https://doi.org/10.21105/joss.02505)
# - R. Gao, M. Deistler, J. H. Macke, *Generalized Bayesian Inference for Scientific Simulators via Amortized Cost Estimation*, NeurIPS (2024). [arXiv:2305.15208](https://arxiv.org/abs/2305.15208)
