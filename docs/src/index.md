# MonteCarloX.jl

MonteCarloX.jl is a modular Monte Carlo framework in Julia.
It separates the sampling algorithm from the system under study:
the user defines the system state and proposes changes; MonteCarloX provides the acceptance criterion.
Because the algorithm is independent of the model, every simulation becomes a template — replacing the system yields a new application without modifying the algorithmic loop.

## Separation of concerns

A Monte Carlo simulation in this framework consists of two parts:

1. **Problem-specific** (provided by the user): the system state and a rule for proposing changes.
2. **Algorithms** (provided by MonteCarloX): the development of the Markov Chain that can vary from simple to complex.

For discrete-state sampling, the central interface is `accept!(algorithm, x_new, x_old)`.
For continuous-time dynamics, the equivalent is `advance!(algorithm, system, T)`, which selects event times and events from user-defined rates.

This separation keeps algorithm code model-agnostic:
the same `Metropolis` algorithm samples a posterior distribution in Bayesian inference, an Ising model at thermal equilibrium, or any other system for which a log-weight function can be defined.

Because the user retains full control over the system definition and update rule, it is straightforward to build companion packages that provide these for entire model families.
For example, `SpinSystems` implements states, updates, and observables for Ising and Blume-Capel models, so that a simulation reduces to choosing an algorithm and running the loop.

## Algorithms

MonteCarloX provides the following sampling algorithms, each usable with any system that implements the required interface:

- **Importance sampling**: `Metropolis`, `Glauber`, `HeatBath` — accept or reject proposed changes based on a log-weight ratio.
- **Flat-histogram methods**: `Multicanonical`, `WangLandau` — iteratively adapt weights to achieve uniform sampling over an order parameter, enabling access to rare configurations.
- **Extended-ensemble methods**: `ParallelTempering`, `ReplicaExchange` — run multiple replicas at different parameters and exchange configurations to overcome free-energy barriers.
- **Continuous-time sampling**: `Gillespie` — exact stochastic simulation via event rates.

## Examples as templates

The documentation includes worked examples across several domains.
Each example is a self-contained simulation that serves as a template:
the algorithmic structure remains unchanged when the system is replaced.

- **Bayesian inference**: posterior sampling for coin flips, linear regression, hierarchical models.
- **Statistical mechanics**: importance sampling, multicanonical sampling, and parallel tempering for Ising and Blume-Capel models.
- **Stochastic processes**: Poisson processes, birth-death dynamics, reversible dimerization via the Gillespie algorithm.
- **Large deviation theory**: multicanonical sampling of rare fluctuations in sums of random variables and the Ornstein-Uhlenbeck process.
- **Infrastructure**: checkpointing and parallel chains (MPI, threads).

## Scope

MonteCarloX provides the algorithmic core.
Concrete model families (e.g., `SpinSystems`) are maintained as separate companion packages.
This keeps the framework compact and allows independent development of new models.

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

