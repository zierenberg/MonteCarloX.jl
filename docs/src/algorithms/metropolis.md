# Metropolis

The Metropolis algorithm is the default first-choice MCMC sampler: propose a local change, compute the log-weight difference, accept with probability ``\min(1, e^{\Delta})``.
For the shared MCMC protocol — `accept!`, ensembles, the linearity trait — see the [class page](markov_chain_monte_carlo.md).

## When to use it

- Standard equilibrium sampling at fixed parameters (canonical ensemble, fixed posterior).
- Whenever the local proposal is *symmetric*: ``q(x' \mid x) = q(x \mid x')``. Asymmetric proposals need Metropolis-Hastings (TODO), which corrects the acceptance with the proposal ratio.
- When the log-weight change is cheap to compute as a local difference (e.g. ``\Delta E`` for a spin flip). Linear ensembles such as `BoltzmannEnsemble` enable this directly via the delta form of `accept!`.

## Acceptance rule

Given current state ``x`` and proposal ``x'`` drawn from a symmetric ``q``, the Metropolis-Hastings acceptance probability reduces to

```math
\alpha(x \to x') = \min(1,\ e^{\Delta}),
\quad
\Delta = \text{logweight}(\text{ens}, x') - \text{logweight}(\text{ens}, x).
```

`accept!` evaluates this rule and updates the algorithm's `steps` and `accepted` counters.

## Constructors

```julia
Metropolis(rng::AbstractRNG; β::Real)                 # convenience: BoltzmannEnsemble(β=β)
Metropolis(rng::AbstractRNG, ens::AbstractEnsemble)   # explicit ensemble
```

The convenience form is the canonical "Metropolis at inverse temperature ``\beta``" call.
The explicit form accepts any `AbstractEnsemble` with `linear_logweight(ens) == true`; non-linear ensembles raise `ArgumentError` at construction.
For non-linear targets (e.g. `MulticanonicalEnsemble`, `WangLandauEnsemble`, or a general `FunctionEnsemble`), use the generic `ImportanceSampling` or one of the dedicated constructors documented on [Multicanonical](multicanonical.md) and [Wang-Landau](wang_landau.md).

## Glauber: logistic acceptance variant

`Glauber` shares the proposal and log-ratio mechanics of `Metropolis` but replaces the Metropolis acceptance rule with the logistic function:

```math
p_{\text{accept}}(\Delta) = \frac{1}{1 + e^{-\Delta}}.
```

Both rules satisfy detailed balance and reach the same stationary distribution; they differ only in the off-equilibrium dynamics and acceptance statistics.
Use `Glauber` when the dynamics being modelled prescribe logistic acceptance (e.g. certain Ising-dynamics conventions, some neuroscience contexts).
Constructors mirror `Metropolis`:

```julia
alg = Glauber(rng; β=0.44)
```

## Example: a self-contained 1D Ising chain

A minimal, dependency-free example that exposes the propose / ``\Delta E`` / accept / commit pattern explicitly.
The "system" here is just a `Vector{Int}` with two helper functions; no companion package is involved.

```julia
using MonteCarloX, Random

const L = 32
spins        = rand([-1, 1], L)
local_dE(s, i) = 2 * s[i] * (s[mod1(i-1, L)] + s[mod1(i+1, L)])

rng = Xoshiro(1)
alg = Metropolis(rng; β=1.0)

for _ in 1:200_000
    i  = rand(rng, 1:L)
    ΔE = local_dE(spins, i)         # log-weight change for flipping spin i
    if accept!(alg, ΔE)              # delta form: Boltzmann is linear
        spins[i] = -spins[i]
    end
end

println("acceptance: ", acceptance_rate(alg))
```

What MonteCarloX provides is `accept!(alg, ΔE)` — the decision.
Everything else — picking a site, computing ``\Delta E``, committing the flip — is system-layer code.
This separation is what makes the algorithm reusable: for a Bayesian posterior, the same `accept!` is called with `(θ_new, θ)`; for a polymer move, with `(config_new, config_old)`.

Companion packages bundle this loop body for their model families.
For richer spin systems (Ising / Blume-Capel on lattices, graphs, or arbitrary couplings), [`MCXSpins`](https://github.com/zierenberg/MonteCarloX.jl/tree/main/MCXSpins) provides the system types and a `spin_flip!(sys, alg)` wrapper that hides the propose / ``\Delta E`` / accept / commit dance:

```julia
using MonteCarloX, MCXSpins, Random
sys = Ising([32, 32]; J=1); init!(sys, :hot; rng=Xoshiro(1))
alg = Metropolis(Xoshiro(1); β=0.44)
for _ in 1:1_000_000
    spin_flip!(sys, alg)
end
```

Replacing `Metropolis` with `Glauber` or `HeatBath` in either example leaves the loop body unchanged; the algorithm type selects the update rule.

For a runnable, plotted version see [Ising 2D (importance sampling)](../generated/importance_Ising2D.md).
The class page shows the corresponding [Bayesian variant](markov_chain_monte_carlo.md) (same loop shape, different ensemble, full-state `accept!`).

## API reference

```@docs
Metropolis
Glauber
AbstractMetropolis
accept!(alg::AbstractMetropolis, delta_arg)
```
