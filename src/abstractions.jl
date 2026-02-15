# Core API Abstractions
# Based on notebooks/api.ipynb
# 
# These abstractions are shared by all Monte Carlo algorithms,
# both equilibrium (Metropolis, etc.) and non-equilibrium (Gillespie, KMC, etc.)

"""
    AbstractSystem

Base type for all systems in MonteCarloX.

A system contains:
- State and constraints
- For equilibrium: spins, particles, configuration
- For non-equilibrium: population, rates, state variables
- In Bayesian case: data and latent variables
- Everything needed to calculate log_weight or rates

Examples:
- Equilibrium: Ising model, particle system
- Non-equilibrium: Birth-death process, SIR model
- Bayesian: Hierarchical model with latent variables
"""
abstract type AbstractSystem end

"""
    AbstractLogWeight

Base type for log weight functions.

A LogWeight is:
- A callable object or function that operates on a system
- May store parameters (e.g., temperature β for Boltzmann)
- May include a measure (for multicanonical, continuous functions, etc.)
- In Bayesian case: log_likelihood + log_prior
"""
abstract type AbstractLogWeight end

"""
    AbstractAlgorithm

Base type for all Monte Carlo algorithms.

An Algorithm:
- Represents the sampling/simulation method
- May hold RNG, parameters, and statistics
- Works with AbstractSystem

Examples:
- Equilibrium: Metropolis, Heat Bath, Cluster updates
- Non-equilibrium: Gillespie, Kinetic Monte Carlo, Poisson process
"""
abstract type AbstractAlgorithm end

"""
    AbstractUpdate

Base type for update methods.

An Update is:
- A function that operates on (System, Algorithm)
- Coordinates how system and logweight interact
- Should be as general as possible for pre-implemented versions
"""
abstract type AbstractUpdate end

"""
    AbstractMeasurement

Base type for measurement objects.
"""
abstract type AbstractMeasurement end

"""
    AbstractImportanceSampling <: AbstractAlgorithm

Base type for importance sampling algorithms (Metropolis, Heat Bath, etc.).

Importance sampling algorithms:
- Use accept/reject steps based on log weight ratios
- Track acceptance statistics
- Include an RNG and a log weight function
"""
abstract type AbstractImportanceSampling <: AbstractAlgorithm end
