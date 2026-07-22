# Metropolis, Glauber, and the balance function

The Metropolis algorithm is the default first-choice MCMC sampler: propose a local change, form the log acceptance-ratio ``\log R``, and accept with probability ``\min(1, e^{\log R})``.
For the shared MCMC protocol — `accept!`, ensembles, the linearity trait — see the [class page](markov_chain_monte_carlo.md).

## When to use it

- Standard equilibrium sampling at fixed parameters (canonical ensemble, fixed posterior).
- Whenever the local proposal is *symmetric*: ``q(x' \mid x) = q(x \mid x')``. Asymmetric proposals are Metropolis-Hastings — no separate type: fold the proposal ratio into ``\log R`` at the call site.
- When the log-weight change is cheap to compute as a local difference (e.g. ``\Delta E`` for a spin flip).

## The acceptance ratio and the balance function

Given current state ``x`` and proposal ``x'``, the log acceptance-ratio is

```math
\log R = \log\frac{\pi(x')\,q(x \mid x')}{\pi(x)\,q(x' \mid x)}
       = \underbrace{\text{logweight}(\text{ens}, \Delta E)}_{\text{ensemble}}
       + \underbrace{\log\frac{q(x \mid x')}{q(x' \mid x)}}_{\text{proposal (0 if symmetric)}}.
```

A **balance function** turns ``\log R`` into dynamics. It is a first-class object in core, shared by every algorithm:

| Balance | acceptance probability ``= `` transition rate | reduces to |
|---------|-----------------------------------------------|------------|
| [`MetropolisBalance`](@ref) | ``\min(1, e^{\log R})`` | Metropolis 1953 / Hastings 1970 |
| [`GlauberBalance`](@ref) | ``1/(1+e^{-\log R}) = \sigma(\log R)`` | Glauber ≡ Barker 1965 |

Both satisfy detailed balance, ``f(\log R)/f(-\log R) = e^{\log R}``, and reach the same stationary distribution; they differ only in the off-equilibrium dynamics.
The single object has **two readings of the same ``f``**:

```julia
acceptance_probability(balance, logR)   # discrete-time reading → accept!
transition_rate(balance, logR)          # continuous-time reading → n-fold / kMC rates
```

This is why "a probability in one model class becomes a rate in another" is literally the same code: the rejection-free n-fold way (`NFoldRates` under `SiteEvents`) builds its event rates from `transition_rate(balance, logR)`, so it is honestly the rejection-free form of the very same `{Metropolis|Glauber}` dynamics.

## Constructors

```julia
MetropolisAlgorithm(rng::AbstractRNG; β::Real)        # convenience: BoltzmannEnsemble(β=β)
MetropolisAlgorithm(rng::AbstractRNG, ens)            # any ensemble
GlauberAlgorithm(rng::AbstractRNG; β::Real)           # Glauber balance
GlauberAlgorithm(rng::AbstractRNG, ens)
```

Both build the [`MetropolisHastingsAlgorithm`](@ref) engine with the appropriate balance; they accept *any* ensemble (linear or not — multicanonical/Wang-Landau are `MetropolisBalance` with an adaptive ensemble).
The short names `Metropolis` / `Glauber` remain as deprecated aliases and resolve with a warning.

## `accept!` takes coordinates; the algorithm owns the target

The algorithm owns the target ``\pi``, so the model hands `accept!` *coordinates*, not a logweight: the algorithm forms ``\log R`` from `ensemble(alg)`. This keeps the *ensemble*, *balance*, and *proposal* concerns separate at the call site. (The raw primitive `accept_logratio!(alg, logR)` sits underneath for a bespoke target not expressed as an ensemble.)

```julia
using MonteCarloX, Random

const L = 32
spins        = rand([-1, 1], L)
local_dE(s, i) = 2 * s[i] * (s[mod1(i-1, L)] + s[mod1(i+1, L)])

rng = Xoshiro(1)
alg = MetropolisAlgorithm(rng; β=1.0)

for _ in 1:200_000
    i  = rand(rng, 1:L)
    ΔE = local_dE(spins, i)                          # energy change for flipping spin i
    if accept!(alg, ΔE)                              # linear ensemble: alg forms logR = -βΔE
        spins[i] = -spins[i]
    end
end

println("acceptance: ", acceptance_rate(alg))
```

`accept!(alg, ΔE)` weights the coordinate difference through the ensemble — ``-\beta\,\Delta E`` for a Boltzmann ensemble, the linear fast path.
For a full-state move, the two-argument form `accept!(alg, arg_new, arg_old)` forms ``\log R`` from `logweight(ens, arg_new) - logweight(ens, arg_old)` (this is the path multicanonical / Wang-Landau use, and it drives their visit recording).
An asymmetric proposal adds a `correction` keyword (the log proposal-ratio) to either form.

Companion packages bundle this loop body. For richer spin systems, [`MCXSpins`](https://github.com/zierenberg/MonteCarloX.jl/tree/main/MCXSpins) provides the system types and a `spin_flip!(sys, alg)` wrapper that hides the propose / ``\Delta E`` / accept / commit dance:

```julia
using MonteCarloX, MCXSpins, Random
sys = IsingSystem([32, 32]; J=1); init!(sys, :random; rng=Xoshiro(1))
alg = MetropolisAlgorithm(Xoshiro(1); β=0.44)
for _ in 1:1_000_000
    spin_flip!(sys, alg)
end
```

Replacing `MetropolisAlgorithm` with `GlauberAlgorithm` or `HeatBathAlgorithm` leaves the loop body unchanged; the algorithm selects the update rule (heat bath draws directly from the local conditional — no accept step).

For a runnable, plotted version see [Ising 2D (MCMC algorithms)](../generated/mcmc_Ising2D.md).

## API reference

```@docs
MetropolisAlgorithm
GlauberAlgorithm
BalanceFunction
MetropolisBalance
GlauberBalance
acceptance_probability
transition_rate
balance
```

Adaptive warm-up for a random-walk proposal — tune the step to a target acceptance
rate from the accept/reject decisions, then freeze it for the sampling phase:

```@docs
AdaptiveStep
step_size
adapt!
```
