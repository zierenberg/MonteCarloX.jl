"""
    AbstractLatticeParticleSystem <: AbstractSystem

Base type for lattice-based particle systems (lattice gas, lattice polymers).

Implementations share cubic lattice geometry with periodic boundary conditions
and precomputed neighbor tables.
"""
abstract type AbstractLatticeParticleSystem <: AbstractSystem end
