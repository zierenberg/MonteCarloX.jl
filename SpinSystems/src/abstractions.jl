"""
    AbstractSpinSystem <: AbstractSystem

Base type for spin systems.

A spin system typically contains:
- Spins/states configuration
- Graph/lattice structure
- Interaction parameters
- Cached quantities for efficient updates
"""
abstract type AbstractSpinSystem <: AbstractSystem end

"""
    pick_site(rng, N)

Randomly pick a site index from 1 to N.
"""
@inline pick_site(rng, N) = rand(rng, UInt) % N + 1
