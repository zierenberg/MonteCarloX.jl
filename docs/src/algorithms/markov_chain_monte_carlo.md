# Markov Chain Monte Carlo

The algorithms in this class are *Markov chain Monte Carlo (MCMC)* samplers: discrete-step Markov chains whose stationary distribution is a target ``\pi(x) \propto \exp(\text{logweight}(x))``, reached either by proposing local changes and accepting or rejecting them, or by drawing one coordinate at a time from its conditional distribution.
The same machinery samples a Boltzmann distribution in statistical mechanics, a posterior in Bayesian inference, or any other density that can be written up to a normalizing constant.
What differs between algorithms is *how* the chain transitions and *what* ensemble defines the target.

This page sets up the vocabulary and API shared by every MCMC algorithm in MonteCarloX.
Concrete algorithms — [Metropolis-Hastings](metropolis.md), Heat bath, [Multicanonical](multicanonical.md), Wang-Landau, Replica exchange — are documented as subpages.

!!! note "Naming"
    `AbstractMarkovChainMonteCarlo` is the umbrella for the whole category. The accept/reject engine is `MetropolisHastingsAlgorithm{E,B,RNG}` (ensemble × balance × rng); `HeatBath` is a sibling concrete type under the *same* umbrella — direct conditional sampling with no accept step, hence no `accepted` counter. The friendly constructors `MetropolisAlgorithm`, `GlauberAlgorithm`, `MulticanonicalAlgorithm`, `WangLandauAlgorithm`, `HeatBathAlgorithm` build these. Old names (`Metropolis`, `Glauber`, `MarkovChainMonteCarlo`, `MCMC`, `AbstractImportanceSampling`, `AbstractMetropolis`, `AbstractHeatBath`) are kept as deprecated bindings and resolve with a warning.

!!! note "Direct & importance sampling"
    Pure i.i.d. sampling from a tractable proposal and standard importance sampling (reweighting i.i.d. draws to a target) are not provided as dedicated algorithm types — they reduce to one-line idioms over two `AbstractEnsemble` objects. Genuinely scalable importance-sampling methods (annealed IS, SMC samplers, population MC, nested sampling, cross-entropy) all live under [Population Monte Carlo](population_monte_carlo.md).

## The ensemble-first framing

A simulation is built from three independent pieces:

| Component | Role | Provided by |
|-----------|------|-------------|
| **System** (`AbstractSystem`) | Holds the state and proposes local changes | User / companion package |
| **Ensemble** (`AbstractEnsemble`) | Defines the target distribution via `logweight(x)` | MonteCarloX or user |
| **Algorithm** (`AbstractMarkovChainMonteCarlo`) | Consumes an ensemble; decides which proposals to accept | MonteCarloX |

In the textbook presentation, an MCMC algorithm is parameterized directly by the physical quantity it samples (e.g. Metropolis with inverse temperature ``\beta``).
MonteCarloX inverts this: the ensemble is the first-class object that defines the target, and algorithms are *consumers* of an ensemble.
The consequences are concrete:

- **Bayesian inference and statistical mechanics share an interface.** `MetropolisAlgorithm(rng, BoltzmannEnsemble(β=1.0))` and `MetropolisAlgorithm(rng, FunctionEnsemble(logposterior))` differ only in the ensemble.
- **Replica exchange is an ensemble swap.** Two algorithms hold two ensembles; a successful exchange moves the ensembles, not the configurations.
- **Adaptive methods are ensembles that learn.** Multicanonical and Wang-Landau are not new algorithms; they are `MetropolisHastingsAlgorithm` (Metropolis balance) with an adaptive ensemble whose `update!` reshapes `logweight` from accumulated histograms.
- **Metropolis vs Glauber is a balance-function choice.** The two differ only in the `BalanceFunction` slot; the same balance also supplies continuous-time rates via `transition_rate` (see [Metropolis, Glauber, and the balance function](metropolis.md)).

## Targets, chains, and the acceptance rule

Each algorithm constructs a Markov chain ``x_0, x_1, \ldots, x_n, \ldots`` on the state space of a system.
Convergence to the target ``\pi`` follows from two conditions: *ergodicity* (the proposal can reach any state with positive probability) and *detailed balance* (the transition probabilities satisfy ``\pi(x) P(x \to x') = \pi(x') P(x' \to x)``).
The system supplies ergodicity through its proposal mechanism; the algorithm supplies detailed balance through its acceptance rule.

For a proposal density ``q(x' \mid x)``, the Metropolis-Hastings acceptance probability is

```math
\alpha(x \to x') = \min\!\left(1,\ \frac{\pi(x')\, q(x \mid x')}{\pi(x)\, q(x' \mid x)}\right).
```

When ``q`` is symmetric, this reduces to ``\pi(x') / \pi(x) = \exp(\text{logweight}(x') - \text{logweight}(x))``.
This log-ratio is what every accept/reject MCMC algorithm in MonteCarloX computes; the algorithm decides how to use it. Conditional samplers (heat bath / Gibbs) bypass the log-ratio entirely by drawing one coordinate at a time from its exact conditional.

## Ensembles

An `AbstractEnsemble` must implement

```julia
logweight(ens::AbstractEnsemble, x) -> Real
```

returning ``\log \pi(x)`` up to an additive constant.
Built-in ensembles:

| Ensemble | `logweight(x)` | Linear? | Typical use |
|----------|----------------|---------|-------------|
| `BoltzmannEnsemble(β=β)` | ``-\beta E`` | yes | Canonical equilibrium sampling |
| `FunctionEnsemble(f; linear=false)` | ``f(x)`` | user-asserted | Bayesian posteriors, arbitrary densities |
| `MulticanonicalEnsemble(bins)` | tabulated ``W(x)`` over bins | no | Flat-histogram sampling across barriers |
| `WangLandauEnsemble(bins)` | tabulated ``W(x)``, modified on every visit | no | On-the-fly density-of-states estimation |

Adaptive ensembles also expose `update!(ens, ...)`, which reshapes `logweight` from accumulated data.
See [Multicanonical](multicanonical.md) and Wang-Landau for the update rules.

### The linearity trait

`linear_logweight(ens)` returns `true` when

```math
\text{logweight}(\text{ens}, \Delta x) = \text{logweight}(\text{ens}, x + \Delta x) - \text{logweight}(\text{ens}, x),
```

i.e. when the log-weight change under a proposal can be computed from a state *difference* alone.
For Boltzmann, ``\Delta \log \pi = -\beta \Delta E``, so the trait holds.
For multicanonical and Wang-Landau, the tabulated weight is non-linear in ``x``, so it does not.

This trait selects which *form* of the log-ratio is cheap — not which algorithm you may build.
For a linear ensemble the log-ratio comes from the local difference alone, ``\log R = \text{logweight}(\text{ens}, \Delta E)`` (the spin fast path).
For a non-linear ensemble (multicanonical, Wang-Landau) the weight is not linear in ``\Delta E``, so the log-ratio is assembled from absolute values around the move via the two-argument `accept!(alg, arg_new, arg_old)`.
Both forms drive the same `MetropolisHastingsAlgorithm` engine; `linear_logweight(ens)` is a compile-time constant, so a system's update can branch on it at no cost.
(Locality of the *weight* — needed by cluster bonds and cached n-fold rates — is a separate factorization property, asserted only where those constructions are built.)

## Algorithms

The accept/reject engine `MetropolisHastingsAlgorithm <: AbstractMarkovChainMonteCarlo` carries:

| Field | Meaning |
|-------|---------|
| `rng::AbstractRNG` | Random number source (any Julia `AbstractRNG`) |
| `ensemble::AbstractEnsemble` | The target distribution |
| `balance::BalanceFunction` | The dynamics (Metropolis / Glauber) |
| `steps::Int` | Cumulative attempted moves |
| `accepted::Int` | Cumulative accepted moves |

Shared API:

```julia
accept!(alg, logR)             -> Bool   # primitive: apply the balance function to logR
accept!(alg, arg_new, arg_old) -> Bool   # convenience: forms logR from a logweight difference
acceptance_rate(alg)           -> Float64
balance(alg)                   -> BalanceFunction
steps(alg)                     -> Int
ensemble(alg)                  -> AbstractEnsemble
reset!(alg)                              # zeroes step and acceptance counters
```

`accept!(alg, logR)` is the core contract: it applies `balance(alg)` to the log acceptance-ratio and updates the counters.
The caller assembles ``\log R`` — for a symmetric move on a linear ensemble that is `logweight(ensemble(alg), ΔE)` (local and cheap for a spin flip); the two-argument convenience forms it from ``\text{logweight}(\text{ens}, x_{\text{new}}) - \text{logweight}(\text{ens}, x_{\text{old}})`` (unavoidable for a full Bayesian posterior move, and the form multicanonical / Wang-Landau use for visit recording).

## A canonical simulation loop

The three pieces compose into a loop whose structure is the same across every algorithm in this class:

```julia
using MonteCarloX, Random

rng = Xoshiro(1)
sys = ...                                          # AbstractSystem (from a companion package)
alg = MetropolisAlgorithm(rng; β=0.44)             # ensemble × balance

measurements = Measurements(
    [:energy => energy => Float64[]],
    interval=100,
)

for step in 1:1_000_000
    i  = rand(rng, 1:nsites(sys))
    ΔE = local_energy_change(sys, i)               # system-defined: change if site i were flipped
    if accept!(alg, logweight(ensemble(alg), ΔE))  # linear fast path: logR = -βΔE
        flip!(sys, i)                              # commit the change
    end
    measure!(measurements, sys, step)
end

println("acceptance: ", acceptance_rate(alg))
```

Swap `MetropolisAlgorithm` for `MulticanonicalAlgorithm` or `WangLandauAlgorithm`, and the loop body needs one change: the ensemble is non-linear, so the system forms the log-ratio from absolute values — pass `(arg_new, arg_old)` to `accept!`.

### Bayesian variant: same loop, different ensemble

A general log-posterior is non-linear in the parameter vector ``\theta``, so the loop forms the log-ratio from the full-state difference (the two-argument `accept!`):

```julia
using MonteCarloX, Random

rng  = Xoshiro(1)
θ    = zeros(D)
alg  = MetropolisAlgorithm(rng, FunctionEnsemble(logposterior))

measurements = Measurements(
    [:logp => logposterior => Float64[]],
    interval=10,
)

for step in 1:1_000_000
    θ_new = θ .+ σ .* randn(rng, D)                # symmetric random-walk proposal
    if accept!(alg, θ_new, θ)                      # full-state form
        θ = θ_new
    end
    measure!(measurements, θ, step)
end
```

The loop *shape* is identical to the stat-mech example: propose, accept-or-reject, measure.
Only the ensemble, the algorithm carrier, and the form of `accept!` differ.

## Choosing an algorithm

| Goal | Algorithm |
|------|-----------|
| Sample equilibrium at fixed temperature or a fixed posterior | [`Metropolis`](metropolis.md) (also: `Glauber`) |
| Update one coordinate at a time from its exact conditional distribution | `HeatBath` (also: Gibbs sampling) |
| Sample across free-energy barriers using pre-computed flat-histogram weights | [`Multicanonical`](multicanonical.md) |
| Estimate the density of states without prior knowledge of the weights | `WangLandau` |
| Overcome critical slowing-down with parallel chains at different parameters | `ReplicaExchange` (also: `ParallelTempering`) |

## Diagnostics

The shared API exposes `acceptance_rate(alg)` for the simplest diagnostic.
Useful ranges depend on the system; for high-dimensional local moves, 30–50% is a common rule of thumb, but the pathologies are visible from either end:

- Very high acceptance (>90%) typically indicates proposals too small to mix efficiently.
- Very low acceptance (<5%) typically indicates proposals too large or the target too peaked.

For adaptive ensembles ([Multicanonical](multicanonical.md), Wang-Landau), additional histogram-based diagnostics (`flatness`, `Roundtrips`) are documented with those algorithms.

For correlation diagnostics, the [measurements](../infrastructure/measurements.md) infrastructure provides `integrated_autocorrelation_time` on recorded observable traces — the effective number of independent samples in a chain of length ``N`` is ``N / (2 \tau_{\text{int}})``:

```julia
τ = integrated_autocorrelation_time(data(measurements, :energy))
```

## API reference

```@docs
AbstractMarkovChainMonteCarlo
MetropolisHastingsAlgorithm
AbstractEnsemble
linear_logweight
logweight(ens::AbstractEnsemble)
logweight(ens::AbstractEnsemble, arg)
ensemble(alg::AbstractMarkovChainMonteCarlo)
accept!
acceptance_rate
reset!(alg::MetropolisHastingsAlgorithm)
steps
update!(ens::AbstractEnsemble, args...)
```

## See also

- [Metropolis-Hastings](metropolis.md) — accept/reject sampler with symmetric proposals (covers `Glauber` as a variant)
- Heat bath — conditional sampling (Gibbs)
- [Multicanonical](multicanonical.md) — flat-histogram sampling with iterative weight refinement (covers `ParallelMulticanonical`)
- Wang-Landau — flat-histogram sampling with on-the-fly weight adaptation
- Replica exchange — parallel chains coupled by ensemble swaps (covers `ParallelTempering`)
- [Population Monte Carlo](population_monte_carlo.md) — home of advanced importance-sampling methods (AIS, SMC, PMC, nested sampling)
