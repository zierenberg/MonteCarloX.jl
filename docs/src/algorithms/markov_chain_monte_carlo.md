# Markov Chain Monte Carlo

The algorithms in this class are *Markov chain Monte Carlo (MCMC)* samplers: discrete-step Markov chains whose stationary distribution is a target ``\pi(x) \propto \exp(\text{logweight}(x))``, reached either by proposing local changes and accepting or rejecting them, or by drawing one coordinate at a time from its conditional distribution.
The same machinery samples a Boltzmann distribution in statistical mechanics, a posterior in Bayesian inference, or any other density that can be written up to a normalizing constant.
What differs between algorithms is *how* the chain transitions and *what* ensemble defines the target.

This page sets up the vocabulary and API shared by every MCMC algorithm in MonteCarloX.
Concrete algorithms — [Metropolis-Hastings](metropolis.md), [Heat bath](heat_bath.md), [Multicanonical](multicanonical.md), [Wang-Landau](wang_landau.md), [Replica exchange](replica_exchange.md) — are documented as subpages.

!!! note "Naming"
    The current code calls the abstract supertype `AbstractImportanceSampling` and the generic algorithm `ImportanceSampling`; both will be renamed to `AbstractMarkovChainMonteCarlo` and `MarkovChainMonteCarlo` in a future release to match the literature. `HeatBath` will also move under this supertype. Page-level vocabulary already follows the new naming.

!!! note "Direct & importance sampling"
    Pure i.i.d. sampling from a tractable proposal and standard importance sampling (reweighting i.i.d. draws to a target) are not provided as dedicated algorithm types — they reduce to one-line idioms over two `AbstractEnsemble` objects. Genuinely scalable importance-sampling methods (annealed IS, SMC samplers, population MC, nested sampling, cross-entropy) all live under [Population Monte Carlo](population_monte_carlo.md).

## The ensemble-first framing

A simulation is built from three independent pieces:

| Component | Role | Provided by |
|-----------|------|-------------|
| **System** (`AbstractSystem`) | Holds the state and proposes local changes | User / companion package |
| **Ensemble** (`AbstractEnsemble`) | Defines the target distribution via `logweight(x)` | MonteCarloX or user |
| **Algorithm** (`AbstractImportanceSampling`) | Consumes an ensemble; decides which proposals to accept | MonteCarloX |

In the textbook presentation, an MCMC algorithm is parameterized directly by the physical quantity it samples (e.g. Metropolis with inverse temperature ``\beta``).
MonteCarloX inverts this: the ensemble is the first-class object that defines the target, and algorithms are *consumers* of an ensemble.
The consequences are concrete:

- **Bayesian inference and statistical mechanics share an interface.** `Metropolis(rng, BoltzmannEnsemble(β=1.0))` and `Metropolis(rng, FunctionEnsemble(logposterior, linear=true))` differ only in the ensemble.
- **Replica exchange is an ensemble swap.** Two algorithms hold two ensembles; a successful exchange moves the ensembles, not the configurations.
- **Adaptive methods are ensembles that learn.** Multicanonical and Wang-Landau are not new algorithms; they are `ImportanceSampling` with an adaptive ensemble whose `update!` reshapes `logweight` from accumulated histograms.

## Targets, chains, and the acceptance rule

Each algorithm constructs a Markov chain ``x_0, x_1, \ldots, x_n, \ldots`` on the state space of a system.
Convergence to the target ``\pi`` follows from two conditions: *ergodicity* (the proposal can reach any state with positive probability) and *detailed balance* (the transition probabilities satisfy ``\pi(x) P(x \to x') = \pi(x') P(x' \to x)``).
The system supplies ergodicity through its proposal mechanism; the algorithm supplies detailed balance through its acceptance rule.

For a proposal density ``q(x' \mid x)``, the Metropolis-Hastings acceptance probability is

```math
\alpha(x \to x') = \min\!\left(1,\ \frac{\pi(x')\, q(x \mid x')}{\pi(x)\, q(x' \mid x)}\right).
```

When ``q`` is symmetric, this reduces to ``\pi(x') / \pi(x) = \exp(\text{logweight}(x') - \text{logweight}(x))``.
This log-ratio is what every accept/reject MCMC algorithm in MonteCarloX computes; the algorithm decides how to use it. Conditional samplers ([heat bath / Gibbs](heat_bath.md)) bypass the log-ratio entirely by drawing one coordinate at a time from its exact conditional.

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
See [Multicanonical](multicanonical.md) and [Wang-Landau](wang_landau.md) for the update rules.

### The linearity trait

`linear_logweight(ens)` returns `true` when

```math
\text{logweight}(\text{ens}, \Delta x) = \text{logweight}(\text{ens}, x + \Delta x) - \text{logweight}(\text{ens}, x),
```

i.e. when the log-weight change under a proposal can be computed from a state *difference* alone.
For Boltzmann, ``\Delta \log \pi = -\beta \Delta E``, so the trait holds.
For multicanonical and Wang-Landau, the tabulated weight is non-linear in ``x``, so it does not.

This trait gates which algorithm form can be used.
Delta-based algorithms ([`Metropolis`](metropolis.md), `Glauber`) require `linear_logweight(ens) == true` and error otherwise:

```julia
Metropolis(rng, MulticanonicalEnsemble(bins))
# ArgumentError: MulticanonicalEnsemble does not have a linear logweight and
# cannot be used with Metropolis. Use ImportanceSampling or a dedicated algorithm instead.
```

For non-linear ensembles use the generic `ImportanceSampling` algorithm — which always evaluates the full-state log-weight difference — or one of the dedicated constructors (`Multicanonical`, `WangLandau`) that wrap it.

## Algorithms

Every accept/reject MCMC algorithm subtypes `AbstractImportanceSampling` (to be renamed `AbstractMarkovChainMonteCarlo`) and carries:

| Field | Meaning |
|-------|---------|
| `rng::AbstractRNG` | Random number source (any Julia `AbstractRNG`) |
| `ensemble::AbstractEnsemble` | The target distribution |
| `steps::Int` | Cumulative attempted moves |
| `accepted::Int` | Cumulative accepted moves |

Shared API:

```julia
accept!(alg, arg_new, arg_old) -> Bool   # full-state form (always available)
accept!(alg, Δx)            -> Bool   # delta form (linear ensembles only)
acceptance_rate(alg)        -> Float64
steps(alg)                  -> Int
ensemble(alg)               -> AbstractEnsemble
reset!(alg)                                # zeroes step and acceptance counters
```

`accept!` updates the internal counters and returns whether the proposal was accepted.
The full-state form computes ``\text{logweight}(\text{ens}, x_{\text{new}}) - \text{logweight}(\text{ens}, x_{\text{old}})``;
the delta form computes ``\text{logweight}(\text{ens}, \Delta x)`` directly and requires linearity.
Either form may be more efficient depending on the system: for a spin flip, ``\Delta E`` is local and cheap;
for a Bayesian posterior move, recomputing ``\log \pi`` over the full parameter vector is usually unavoidable.

## A canonical simulation loop

The three pieces compose into a loop whose structure is the same across every algorithm in this class:

```julia
using MonteCarloX, Random

rng = Xoshiro(1)
sys = ...                                          # AbstractSystem (from a companion package)
alg = Metropolis(rng; β=0.44)                      # algorithm + ensemble

measurements = Measurements(
    [:energy => energy => Float64[]],
    interval=100,
)

for step in 1:1_000_000
    i  = rand(rng, 1:nsites(sys))
    ΔE = local_energy_change(sys, i)               # system-defined: change if site i were flipped
    if accept!(alg, ΔE)                            # delta form (Boltzmann is linear)
        flip!(sys, i)                              # commit the change
    end
    measure!(measurements, sys, step)
end

println("acceptance: ", acceptance_rate(alg))
```

Swap `Metropolis` for `Multicanonical` or `WangLandau`, and the loop body needs one change: the delta form is no longer available, so the system passes `(arg_new, arg_old)` to `accept!`.

### Bayesian variant: same loop, different ensemble

A general log-posterior is non-linear in the parameter vector ``\theta``, so we use the generic `ImportanceSampling` algorithm (which always computes the full-state log-ratio) and the corresponding form of `accept!`:

```julia
using MonteCarloX, Random

rng  = Xoshiro(1)
θ    = zeros(D)
alg  = ImportanceSampling(rng, FunctionEnsemble(logposterior))   # linear=false (default)

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
| Update one coordinate at a time from its exact conditional distribution | [`HeatBath`](heat_bath.md) (also: Gibbs sampling) |
| Sample across free-energy barriers using pre-computed flat-histogram weights | [`Multicanonical`](multicanonical.md) |
| Estimate the density of states without prior knowledge of the weights | [`WangLandau`](wang_landau.md) |
| Overcome critical slowing-down with parallel chains at different parameters | [`ReplicaExchange`](replica_exchange.md) (also: `ParallelTempering`) |

## Diagnostics

The shared API exposes `acceptance_rate(alg)` for the simplest diagnostic.
Useful ranges depend on the system; for high-dimensional local moves, 30–50% is a common rule of thumb, but the pathologies are visible from either end:

- Very high acceptance (>90%) typically indicates proposals too small to mix efficiently.
- Very low acceptance (<5%) typically indicates proposals too large or the target too peaked.

For adaptive ensembles ([Multicanonical](multicanonical.md), [Wang-Landau](wang_landau.md)), additional histogram-based diagnostics (`flatness`, `Roundtrips`) are documented with those algorithms.

For correlation diagnostics, the [measurements](../measurements.md) infrastructure provides `integrated_autocorrelation_time` on recorded observable traces — the effective number of independent samples in a chain of length ``N`` is ``N / (2 \tau_{\text{int}})``:

```julia
τ = integrated_autocorrelation_time(data(measurements, :energy))
```

## API reference

```@docs
AbstractImportanceSampling
ImportanceSampling
AbstractEnsemble
linear_logweight
logweight(ens::AbstractEnsemble)
logweight(ens::AbstractEnsemble, x)
ensemble(alg::AbstractImportanceSampling)
accept!
acceptance_rate
reset!(alg::AbstractImportanceSampling)
steps
update!(ens::AbstractEnsemble, args...)
```

## See also

- [Metropolis-Hastings](metropolis.md) — accept/reject sampler with symmetric proposals (covers `Glauber` as a variant)
- [Heat bath](heat_bath.md) — conditional sampling (Gibbs)
- [Multicanonical](multicanonical.md) — flat-histogram sampling with iterative weight refinement (covers `ParallelMulticanonical`)
- [Wang-Landau](wang_landau.md) — flat-histogram sampling with on-the-fly weight adaptation
- [Replica exchange](replica_exchange.md) — parallel chains coupled by ensemble swaps (covers `ParallelTempering`)
- [Population Monte Carlo](population_monte_carlo.md) — home of advanced importance-sampling methods (AIS, SMC, PMC, nested sampling)
