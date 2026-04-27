"""
    AbstractSoftMatterSystem <: AbstractSystem

Base type for off-lattice soft matter systems.
"""
abstract type AbstractSoftMatterSystem <: AbstractSystem end

"""
    AbstractPairPotential

Base type for pair potentials V(r²) between two particles.

Implementations must define:
- `(pot::MyPotential)(r_sq)` -- evaluate potential at squared distance r²
- `cutoff_sq(pot)` -- squared cutoff distance (Inf if no cutoff)
"""
abstract type AbstractPairPotential end

"""
    AbstractBondPotential

Base type for bond potentials V(r) between bonded neighbors along a chain.

Implementations must define:
- `(pot::MyBondPotential)(r_sq)` -- evaluate potential at squared distance r²
"""
abstract type AbstractBondPotential end

"""
    AbstractBendingPotential

Base type for bending potentials V(cos θ) at chain angles.

Implementations must define:
- `(pot::MyBendingPotential)(cos_theta)` -- evaluate at cosine of bond angle
"""
abstract type AbstractBendingPotential end

"""
    NoPotential

Zero potential placeholder. Used when a system doesn't need a particular
interaction type (e.g., no bending for flexible polymers).
"""
struct NoPotential <: AbstractPairPotential end
(::NoPotential)(r_sq) = 0.0
cutoff_sq(::NoPotential) = Inf

struct NoBondPotential <: AbstractBondPotential end
(::NoBondPotential)(r_sq) = 0.0

struct NoBendingPotential <: AbstractBendingPotential end
(::NoBendingPotential)(cos_theta) = 0.0
