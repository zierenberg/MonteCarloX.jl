# MonteCarloX.jl

MonteCarloX is a compact Julia framework for Monte Carlo sampling.
It is built around composable primitives, so algorithms stay reusable across very different models.

## What this package gives you

- Equilibrium samplers (`Metropolis`, `Glauber`, `HeatBath`, `Multicanonical`, `WangLandau`)
- Continuous-time samplers (`Gillespie`)
- Measurement/scheduling framework (`Measurements`)
- Log-weight tools (`BoltzmannLogWeight`, `BinnedLogWeight`)
- Event handler backends for event-driven dynamics

## Core idea

A simulation needs 3 pieces:

1. **System**: state and model-specific operations
2. **Weight/rates**: target distribution or transition intensities
3. **Algorithm**: transition sampler

`Measurements` are optional convenience tools for organized observable collection.

This separation keeps algorithm code model-agnostic.

## Reading path

If you are new, read in this order:

1. Monte Carlo Fundamentals
2. Importance Sampling Algorithms
3. Continuous-Time Sampling Algorithms
4. Build Your Own System
5. Systems
6. Weights
7. Measurements


## Quick orientation

- Use **importance sampling** for discrete-step update protocols (equilibrium or driven/non-equilibrium).
- Use **continuous-time sampling** when physical/simulation time matters.
- Use **companion model packages** (for example `SpinSystems`) for concrete systems.

## Scope

MonteCarloX is the algorithmic core. Concrete model families are intentionally external.
This keeps the framework concise and easier to extend.

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
- Uncertainty propagation with particle arithmetic:
	- [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl)

MonteCarloX stays focused on compact, model-agnostic algorithmic building blocks, while these packages offer specialized ecosystems for their target domains.

## Random number generators

MonteCarloX works with any Julia `AbstractRNG`.

- Prefer `Xoshiro` as a modern default for new projects.
- Use `MersenneTwister` when compatibility with existing workflows is needed.

Because RNG is passed directly to algorithms, changing RNG is a one-line change.

