# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src

# # Hierarchical Bayesian Inference: The Eight Schools Problem
#
# The eight schools problem [(Rubin, 1981)](https://doi.org/10.2307/1164617)
# is a canonical example of hierarchical Bayesian modelling. A test-preparation
# program was run independently in 8 schools; each school measured a mean
# score improvement ``y_n`` with standard error ``\sigma_n``. The effects
# vary widely and their standard errors are large — so neither pooling all
# schools nor treating them independently is appropriate.
#
# The hierarchical solution: assume each school's true improvement
# ``\theta_n`` is drawn from a shared population distribution, so information
# is *partially pooled* across schools.
#
# ```math
# \theta_n \sim \mathcal{N}(\mu, \tau), \qquad
# y_n \sim \mathcal{N}(\theta_n, \sigma_n)
# ```
#
# with weakly informative priors ``\mu \sim \mathcal{N}(0,5)`` and
# ``\log\tau \sim \mathcal{N}(0,5)``.

using Random, Statistics, Distributions, Plots, StatsBase
using MonteCarloX

# ## Data

y = [28.0,  8.0, -3.0,  7.0, -1.0,  1.0, 18.0, 12.0]  ## mean score improvements
σ = [15.0, 10.0, 16.0, 11.0,  9.0, 11.0, 10.0, 18.0]  ## standard errors
J = length(y)

println("School effects: ", y)
println("Standard errors: ", σ)

# ## Model
#
# We follow the **TensorFlow Probability** parameterisation
# [(TFP eight schools)](https://www.tensorflow.org/probability/examples/Eight_Schools).
# To keep the sampler in an unconstrained space, we sample ``\log\tau``
# directly rather than ``\tau``, with a Normal prior on ``\log\tau``:
#
# ```math
# \mu \sim \mathcal{N}(0, 10), \qquad
# \log\tau \sim \mathcal{N}(5, 1)
# ```
#
# This is equivalent to a LogNormal prior on ``\tau``, which keeps ``\tau``
# strictly positive by construction — no constraint or Jacobian needed. 
# The prior ``\log\tau \sim \mathcal{N}(5,1)`` places most mass on
# ``\tau \in [e^4, e^6] \approx [55, 400]``, i.e. weakly informative
# toward larger between-school variation.
# This follows the **TensorFlow Probability** implementation of the
# eight schools problem, see
# [Eight Schools (TFP)](https://www.tensorflow.org/probability/examples/Eight_Schools).
#
# The **PyMC/Stan** canonical version instead uses
# ``\tau \sim \text{HalfCauchy}(0, 5)``, which has heavier tails and
# places more mass near zero — favouring stronger pooling. Results will
# differ slightly between the two prior choices.
#
# The full state vector is ``[\mu,\, \log\tau,\, \theta_1,\ldots,\theta_8]``.

get_μ(s)    = s[1]
get_logτ(s) = s[2]
get_θ(s)    = s[3:end]

function logposterior(s)
    μ, logτ, θ = get_μ(s), get_logτ(s), get_θ(s)
    τ = exp(logτ)
    lp  = logpdf(Normal(0, 10), μ)      ## population mean effect
    lp += logpdf(Normal(5, 1), logτ)    ## log τ ~ N(5,1)  ↔  τ ~ LogNormal(5,1)
    lp += sum(logpdf.(Normal(μ, τ), θ)) ## school effects drawn from population
    lp += sum(logpdf.(Normal.(θ, σ), y))  ## observed scores
    return lp
end

s0 = [mean(y); log(std(y)); copy(y)];

## initialise at the data
s0 = [mean(y); log(std(y)); copy(y)]
println("Initial log-posterior: ", round(logposterior(s0); digits=2))

# ## Update function
# The log-posterior is evaluated at the proposed state and the current state
# to compute the acceptance probability. The proposal is symmetric, so the proposal densities
# cancel out in the Metropolis acceptance ratio.

function update!(s, alg, Δ)
    i = rand(alg.rng, 1:length(s))
    s_new = copy(s)
    s_new[i] += rand(alg.rng, Uniform(-Δ[i], Δ[i]))
    if accept!(alg, s_new, s)
        s[i] = s_new[i]
    end
end

# ## Metropolis sampling
#
# We update one parameter at a time with a uniform random-walk proposal.
# The step sizes are tuned separately for the hyperparameters ``(\mu, \log\tau)``
# and the school effects ``\theta_n`` to achieve reasonable acceptance rates.

function run_metropolis(logposterior, s0;
                        seed=42, n_burn=10_000, n_samples=100_000, thin=10)
    rng     = Xoshiro(seed)
    alg     = Metropolis(rng, logposterior)
    s       = copy(s0)
    Δ       = vcat([2.0, 2.0], fill(5.0, length(s)-2)) 
    samples = zeros(length(s), n_samples)

    for _ in 1:n_burn
        update!(s, alg, Δ)
    end

    for k in 1:n_samples*thin
        update!(s, alg, Δ)
        k % thin == 0 && (samples[:, k÷thin] = s)
    end
    return samples, alg
end

samples, alg = run_metropolis(logposterior, s0)
println("Acceptance rate : ", round(acceptance_rate(alg); digits=3))
println("Samples collected : ", size(samples, 2))

# ## Convergence check
#
# The log-posterior trace should be stationary with no visible trend.

lps = [logposterior(samples[:, i]) for i in 1:size(samples,2)]
plot(lps; xlabel="Iteration", ylabel="Log-posterior",
     title="Convergence trace", size=(700, 220),
     margin=5Plots.mm, legend=false)

# ## Results
#
# Posterior medians and 95% credible intervals for all parameters.
# Partial pooling shrinks extreme school effects toward the population mean.

μ_s = samples[1, :]
τ_s = exp.(samples[2, :])
θ_s = [samples[2+i, :] for i in 1:J]

println("Parameter   median     95% CI")
println("─────────────────────────────────────")
for (name, s) in vcat([("μ", μ_s), ("τ", τ_s)],
                       [("θ[$(i)]", θ_s[i]) for i in 1:J])
    lo, hi = round.(quantile(s, [0.025, 0.975]); digits=1)
    println(rpad(name, 10), rpad(round(median(s); digits=1), 10),
            "[$(lo), $(hi)]")
end

# ## Posterior marginals
#
# Each panel shows the posterior (histogram) against the prior (curve).
# The hierarchical prior on ``\theta_n`` uses the posterior mean of
# ``(\mu, \tau)`` as its parameters.
using LinearAlgebra
function marginal_panel(s, prior, xlabel)
    hist = fit(Histogram, s; nbins=40)
    dist = normalize(hist; mode=:pdf)
    xs   = range(hist.edges[1][1], hist.edges[1][end]; length=300)
    fig  = plot(dist; st=:bar, linewidth=0, alpha=0.6, label="posterior",
                xlabel=xlabel, ylabel="density")
    plot!(fig, xs, pdf.(prior, xs); lw=2, color=:black, label="prior")
    return fig
end

θ_prior = Normal(median(μ_s), median(τ_s))

panels = [
    marginal_panel(μ_s, Normal(0, 5),   "μ"),
    marginal_panel(τ_s, LogNormal(5, 1), "τ")
]
for i in 1:J
    push!(panels, marginal_panel(θ_s[i], θ_prior, "θ[$(i)]"))
end

plot(panels...; layout=(2, 5), size=(1100, 380), margin=4Plots.mm,
     plot_title="Posterior marginals vs priors")
