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

Here's the trimmed version:

---

So far, we distinguish four main classes of Monte-Carlo sampling schemes that bring different types of algorithms:

- **Markov-Chain Monte Carlo**: discrete-step dynamics based on proposing a new state `x_new` and accepting/rejecting with `accept!(alg, x_new, x_old)`.
  - `Metropolis` — accept or reject proposed changes based on a log-weight ratio. Specializations: `Glauber`, `HeatBath`, `MetropolisHastings` (TODO)
  - `ReplicaExchange` — run multiple replicas at different parameters and exchange configurations to bypass free-energy barriers. Specializations: `ParallelTempering`
  - `Multicanonical` — flat-histogram algorithm where weights are iteratively adapted to achieve uniform sampling of a reaction coordinate across barriers. (TODO: merge `Muca`, `WangLandau`, and `SAMC` into a unified flat-histogram class)
  - `WangLandau` — flat-histogram algorithm where weights are adapted on-the-fly with the same purpose as `Multicanonical`. Specialization: `stochastic approximation Monte Carlo (SAMC)` (TODO)
  - `HMC(TODO)` — gradient-informed Hamiltonian proposals with Metropolis-Hastings correction (needs information about the gradient, not trivial)

- **Kinetic Monte Carlo**: continuous-time dynamics on a discrete state space, governed by a master equation. Based on event sources `es` (e.g. list of rates) to draw `(time, event)` via `step!(alg, es)` or advance the full system via `advance!(alg, sys)` if `event_source(sys)` is specified.
  - `Gillespie` — exact stochastic simulation via event rates
  - `Poisson` — sample (inhomogeneous) Poisson processes

- **Stochastic Dynamics**: continuous-time dynamics on a continuous state space, governed by a Langevin SDE / Fokker-Planck equation. 
  - TODO: Implement as a thin wrapper around [`StochasticDiffEq.jl`](https://github.com/SciML/StochasticDiffEq.jl), defining drift `f(u,p,t)` and diffusion `g(u,p,t)` coefficients.

- **Population Monte Carlo** (sequential MC | particle filter): a population of states (sometimes called particles) that evolve according to some propagation rule and are resampled via `resample!(alg, population)`. Sits orthogonally to the other three classes — any of the above can serve as the within-population propagation kernel.
  - `PopulationAnnealing` — (TODO)
  - `PERM` — (TODO)

The different sampling schemes can be combined. For example, population resampling schemes are often combined with local MCMC equilibration. Similarly, we can also perform MCMC on the random numbers underlying kineticMC to sample rare events.

## Infrastructure

The package comes with helpful infrastructure for advanced Monte Carlo algorithms including
- `BinnedObjects` - object for storing binned weight functions or histograms, both for discrete and continous binning.
- `ParallelBackends` - backends for parallel computing, includes `ThreadsBackend` and `MPIBackend`.
- `ParallelChains` - handles parallel algorithms for MCMC (TODO: may need refactor if we do not consider PopMC Markov Chain).
- `CheckpointSession` - uses Serialization to checkpoint and recover simulations
- `Monitoring` - monitoring tools, inlcluding `RoundTrips`, and histogram flatness criteria
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

- [GeneralizedMonteCarlo.jl](https://juliapackages.com/p/generalizedmontecarlo): generalized-ensemble methods (for example multicanonical and related workflows).
- [MonteCarlo.jl](https://github.com/carstenbauer/MonteCarlo.jl): quantum many-body focused Monte Carlo framework.
- [Carlo.jl](https://github.com/lukas-weber/Carlo.jl): lattice-model Monte Carlo toolkit with a strong focus on physics applications.

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
  - [SpinMC](https://github.com/fbuessen/SpinMC.jl)
- Uncertainty propagation with particle arithmetic:
  - [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl)

MonteCarloX stays focused on compact, model-agnostic algorithmic building blocks, while these packages offer specialized ecosystems for their target domains.

## Random number generators

MonteCarloX works with any Julia `AbstractRNG`.

- Prefer `Xoshiro` as a modern default for new projects.
- Use `MersenneTwister` when compatibility with existing workflows is needed.

Because RNG is passed directly to algorithms, changing RNG is a one-line change.

