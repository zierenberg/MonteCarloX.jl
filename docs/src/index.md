# MonteCarloX.jl

MonteCarloX.jl is a modular Monte Carlo framework in Julia.
It separates the sampling algorithm from the system under study:
the user defines the system state and proposes changes; MonteCarloX provides the acceptance criterion.
By explicitly separating the algorithm from the system, every simulation becomes a template — replacing the system yields a new application without modifying the algorithmic loop.

## Separation of concerns

A Monte Carlo simulation in this framework consists of two parts:

1. **Problem-specific** (provided by the user/community): the system state and a rule for proposing changes.
2. **Algorithms** (provided by MonteCarloX): the development of the Markov Chain that can vary from simple to complex.

This separation keeps algorithm code model-agnostic:
We can use the same MCMC algorithm to sample the posterior distribution in Bayesian inference, the Boltzmann distribution of an Ising model, or any other system/ for that a weight function can be defined.

Because the user retains full control over the system definition and update rule, it is straightforward to build companion packages that provide these for entire model families.
For example, `MCXSpins` implements states, updates, and observables for Ising and Blume-Capel models, so that a simulation reduces to choosing an algorithm and running the loop.

## Scope

MonteCarloX provides the algorithmic core.
Concrete model families (e.g., `MCXSpins`) are maintained as separate companion packages.
This keeps the framework compact and allows independent development of new models.

## Algorithms

Sorted by the standard literature partition, MonteCarloX organizes algorithms into method classes:

- **[Basic Sampling](algorithms/basic_sampling.md)** — simple (i.i.d.) sampling and standard importance sampling, expressed as one-line idioms over `AbstractEnsemble` objects. These have no dedicated algorithm type; their genuinely scalable variants are population-based and live under Population Monte Carlo.

- **[Markov Chain Monte Carlo](algorithms/markov_chain_monte_carlo.md)** — discrete-step chains whose transitions follow a local rule: accept/reject from a log-weight difference, or resample one coordinate from its conditional. When that rule derives from a global log-weight (detailed balance), the stationary distribution is the target ``\pi(x) \propto \exp(\text{logweight}(x))``; when only a local difference is defined (e.g. nonreciprocal couplings), detailed balance breaks and the chain instead converges to a nonequilibrium steady state.
  - [`MetropolisAlgorithm`, `GlauberAlgorithm`](algorithms/metropolis.md) — accept/reject with a symmetric proposal; the two differ only in the **balance function** (`MetropolisBalance` vs `GlauberBalance`), which also supplies the continuous-time rates of the rejection-free n-fold way. Both build the `MetropolisHastingsAlgorithm` engine; Metropolis–Hastings needs no separate type — an asymmetric proposal just adds its ratio to ``\log R``.
  - `HeatBathAlgorithm` — Gibbs-style sampling of one coordinate from its conditional (Bayesian "Gibbs sampling"). Subpage TODO.
  - [`MulticanonicalAlgorithm`](algorithms/multicanonical.md), `WangLandauAlgorithm` (subpage TODO), `SAMC` (TODO) — flat-histogram methods (Metropolis balance with an iteratively adapted ensemble).
  - `ReplicaExchange`, `ParallelTempering` — parallel chains coupled by ensemble swaps. Subpage TODO.
  - `HMC` (TODO) — gradient-informed Hamiltonian proposals.

- **[Kinetic Monte Carlo](algorithms/kinetic_monte_carlo.md)** — continuous-time dynamics on a discrete state space, governed by a master equation. Draws `(time, event)` from a rate-based event source.
  - `Gillespie` — exact stochastic simulation via event rates.
  - `Poisson` — primitives for (inhomogeneous) Poisson processes.
  - `BKL` / `n-fold way` (TODO) — discrete-time projection of `Gillespie`.

- **[Population Monte Carlo](algorithms/population_monte_carlo.md)** (planned class, stub) — a weighted particle ensemble evolved through a sequence of intermediate targets. Sits orthogonally to MCMC and KMC, with any of them serving as the per-particle mutation kernel. Future home of all advanced importance-sampling methods (annealed IS, SMC samplers, population annealing, PERM, nested sampling, cross-entropy).

**Continuous-state stochastic dynamics** (Langevin SDE / Fokker-Planck) are out of scope for the core package and can be handled by [`StochasticDiffEq.jl`](https://github.com/SciML/StochasticDiffEq.jl); we may add a thin wrapper later if motivated by a concrete use case.

The classes can be combined. Population MC uses MCMC as a per-particle kernel; MCMC on the random numbers underlying a KMC trajectory enables rare-event sampling of stochastic dynamics.

## Infrastructure

The package comes with helpful infrastructure for advanced Monte Carlo algorithms including
- `BinnedObjects` - object for storing binned weight functions or histograms, both for discrete and continuous binning.
- `ParallelBackends` - backends for parallel computing, includes `ThreadsBackend` and `MPIBackend`.
- `ParallelChains` - handles parallel algorithms for MCMC (TODO: may need refactor if we do not consider PopMC Markov Chain).
- `CheckpointSession` - uses Serialization to checkpoint and recover simulations
- `Monitoring` - monitoring tools, including `RoundTrips`, and histogram flatness criteria
- `MutableRandomNumbers` - useful to update random numbers themselves for rare-event sampling. (TODO: needs to be included into actual examples, so far we just did that manually, so not sure if this is really helping or confusing)

## Examples as templates

The documentation includes worked examples across several domains.
Each example is a self-contained simulation that serves as a template:
the algorithmic structure remains unchanged when the system is replaced.

- **Bayesian inference**: posterior sampling for coin flips, linear regression, hierarchical models.
- **Statistical mechanics**: importance sampling, multicanonical sampling, and parallel tempering for Ising and Blume-Capel models.
- **Stochastic processes**: Poisson processes, birth-death dynamics, reversible dimerization via the Gillespie algorithm.
- **Large deviation theory**: multicanonical sampling of rare fluctuations in sums of random variables and the Ornstein-Uhlenbeck process.
- **Infrastructure**: checkpointing and parallel chains (MPI, threads).

## Related Julia packages

The Julia ecosystem already has several Monte Carlo packages with different goals and interfaces.
If your use case is better served by a domain-specific implementation, these are useful alternatives or complements:

- [MonteCarlo.jl](https://github.com/carstenbauer/MonteCarlo.jl): quantum many-body focused Monte Carlo framework.
- [Carlo.jl](https://github.com/lukas-weber/Carlo.jl): lattice-model Monte Carlo toolkit with a strong focus on physics applications.
- [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl): classical Monte Carlo for lattice spin models, with parallel tempering.

Other related packages (grouped by common use cases):

- Bayesian inference and MCMC:
  - [Turing.jl](https://github.com/TuringLang/Turing.jl)
  - [AbstractMCMC.jl](https://github.com/TuringLang/AbstractMCMC.jl)
  - [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl)
  - [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)
  - [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl)
  - [BAT.jl](https://github.com/bat/BAT.jl)
  - [Gen.jl](https://github.com/probcomp/Gen.jl)
- Monte Carlo integration and low-discrepancy sampling:
  - [Cuba.jl](https://github.com/giordano/Cuba.jl)
  - [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl)
- Uncertainty propagation with particle arithmetic:
  - [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl)

MonteCarloX stays focused on compact, model-agnostic algorithmic building blocks, while these packages offer specialized ecosystems for their target domains.

## Random number generators

MonteCarloX works with any Julia `AbstractRNG`.

- Prefer `Xoshiro` as a modern default for new projects.
- Use `MersenneTwister` when compatibility with existing workflows is needed.
- Use `MutableRandomNumbers` — a custom `AbstractRNG` shipped with MonteCarloX that stores its draws in a mutable vector — when you need to control or perturb the underlying random numbers directly, as in rare-event sampling of stochastic dynamics.

Because RNG is passed directly to algorithms, changing RNG is a one-line change.

