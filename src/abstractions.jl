# Core API Abstractions
# Based on examples/api.ipynb

"""
    AbstractSystem

Base type for all systems in MonteCarloX.

A system contains:
- State and constraints
- In Bayesian case: data and latent variables
- Everything needed to calculate log_weight
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
- Holds the logweight function that operates on the system
- Represents "the" Markov chain
- May store state like RNG, step counters, acceptance statistics
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
