"""
    AbstractPairPotential

Pair potential V(r²) between two particles.

Implementations must define:
- `(pot::MyPotential)(r_sq)` -- evaluate at squared distance
- `cutoff_sq(pot)` -- squared cutoff distance (Inf if no cutoff)
"""
abstract type AbstractPairPotential end

"""
    AbstractBondPotential

Bond potential V(r²) between bonded neighbors along a chain.

Implementations must define:
- `(pot::MyBondPotential)(r_sq)` -- evaluate at squared distance
"""
abstract type AbstractBondPotential end

"""
    AbstractBendingPotential

Bending potential V(cos theta) at chain angles.

Implementations must define:
- `(pot::MyBendingPotential)(cos_theta)` -- evaluate at cosine of bond angle
"""
abstract type AbstractBendingPotential end

# ── Zero placeholders ────────────────────────────────────────────────────────

struct NoPotential <: AbstractPairPotential end
(::NoPotential)(r_sq) = 0.0
cutoff_sq(::NoPotential) = Inf

struct NoBondPotential <: AbstractBondPotential end
(::NoBondPotential)(r_sq) = 0.0

struct NoBendingPotential <: AbstractBendingPotential end
(::NoBendingPotential)(cos_theta) = 0.0
